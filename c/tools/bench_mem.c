/* Memory and storage ceilings. Decode at 50 tok/s needs ~31 GB/s of weight
 * reads (622 MB/token at 2-bit with an 8-bit lm_head), so the question is
 * whether that must come from RAM (and if RAM can supply it) or whether NVMe
 * can carry part of it.
 *
 * Also measures mmap+MADV_WILLNEED vs pread, because the current engine does
 * explicit pread into malloc'd slots; if mmap of a page-cached file is close to
 * memcpy speed, the whole expert-streaming design can change. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <omp.h>

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

int main(int argc,char**argv){
    size_t N=(size_t)2<<30;                    /* 2 GiB */
    char *a=malloc(N),*b=malloc(N);
    memset(a,1,N); memset(b,2,N);

    double t0=now(); memcpy(b,a,N); double dt=now()-t0;
    printf("memcpy 1T        %6.1f GB/s\n",2.0*N/dt/1e9);

    t0=now();
    #pragma omp parallel
    { int nt=omp_get_num_threads(),id=omp_get_thread_num();
      size_t c=N/nt; memcpy(b+id*c,a+id*c,c); }
    dt=now()-t0;
    printf("memcpy %2dT       %6.1f GB/s\n",omp_get_max_threads(),2.0*N/dt/1e9);

    /* pure read bandwidth: sum bytes, no writes */
    t0=now(); volatile uint64_t s=0;
    #pragma omp parallel reduction(+:s)
    { int nt=omp_get_num_threads(),id=omp_get_thread_num();
      size_t c=N/nt/8; const uint64_t *p=(const uint64_t*)(a+id*(N/nt));
      uint64_t l=0; for(size_t i=0;i<c;i++) l+=p[i]; s+=l; }
    dt=now()-t0;
    printf("read   %2dT       %6.1f GB/s  (sum=%llu)\n",omp_get_max_threads(),(double)N/dt/1e9,(unsigned long long)s);

    if(argc>1){
        const char *path=argv[1];
        struct stat st; if(stat(path,&st)){perror("stat");return 1;}
        size_t fsz=st.st_size>(off_t)(1<<30)?(size_t)1<<30:(size_t)st.st_size;

        /* cold-ish pread */
        int fd=open(path,O_RDONLY);
        void *buf=malloc(fsz);
        t0=now();
        size_t off=0; while(off<fsz){ ssize_t r=pread(fd,(char*)buf+off,fsz-off,off); if(r<=0)break; off+=r; }
        dt=now()-t0;
        printf("pread  1T        %6.1f GB/s  (%zu MB, page cache warm/cold mix)\n",(double)fsz/dt/1e9,fsz>>20);

        /* parallel pread: NVMe needs queue depth to reach peak */
        t0=now();
        #pragma omp parallel
        { int nt=omp_get_num_threads(),id=omp_get_thread_num();
          size_t c=fsz/nt, o=id*c;
          int f2=open(path,O_RDONLY);
          size_t d=0; while(d<c){ ssize_t r=pread(f2,(char*)buf+o+d,c-d,o+d); if(r<=0)break; d+=r; }
          close(f2); }
        dt=now()-t0;
        printf("pread  %2dT      %6.1f GB/s  (queue depth matters on NVMe)\n",omp_get_max_threads(),(double)fsz/dt/1e9);

        /* mmap + sequential touch */
        void *mp=mmap(NULL,fsz,PROT_READ,MAP_PRIVATE,fd,0);
        if(mp!=MAP_FAILED){
            madvise(mp,fsz,MADV_WILLNEED);
            t0=now(); uint64_t acc=0;
            #pragma omp parallel reduction(+:acc)
            { int nt=omp_get_num_threads(),id=omp_get_thread_num();
              size_t c=fsz/nt/8; const uint64_t *p=(const uint64_t*)((char*)mp+id*(fsz/nt));
              uint64_t l=0; for(size_t i=0;i<c;i++) l+=p[i]; acc+=l; }
            dt=now()-t0;
            printf("mmap   %2dT      %6.1f GB/s  (acc=%llu)\n",omp_get_max_threads(),(double)fsz/dt/1e9,(unsigned long long)acc);
            munmap(mp,fsz);
        }
        close(fd); free(buf);
    }
    return 0;
}
