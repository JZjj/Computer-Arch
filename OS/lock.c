//如何降低锁竞争 如何处理死锁

//memory allocator
/*
freelist作为空闲物理页面作为链表项，空闲链成为链表。分配物理页就是freelist从链表移除，
释放物理页就是把要释放的页放回链表


*/
struct {
    struct spinlock lock;
    struct run *freelist;
} kmem;

void kfree(void *pa){
    struct run *r;
    if (((uint64)pa % PGSIZE) != 0 || (char*) pa < end || (uint64) pa >= PHYSTOP)
        panic("kfree");
    memset(pa, 1, PGSIZE);

    r = (struct run*) pa;
    acquire(&kmem.lock);
    r->next = kmem.freelist;
    kmem.freelist = r;
    release(&kmem.lock);

}

void* kalloc(void){
    struct run *r;
    acquire(&kmem.lock);
    r = kmem.freelist;
    if (r){
        kmem.freelist = r->next;
    }
    release(&kmem.lock);

    if (r){
        memset((char*)r, 5, PGSIZE);
    }
    return (void*) r;
}

//同一时刻只能一个线程申请或者分配释放内存，多线程无法操作，效率低
//lock cpu

struct{
    struct spinlock lock;
    struct run* freelist;
}kmem[NCPU];

char* kmem_lock_names[] = {
    "kmem_cpu_0",
    "kmem_cpu_1",
    "kmem_cpu_2",
    "kmem_cpu_3",
    "kmem_cpu_4",
    "kmem_cpu_5",
};

void kinit(){
    for (int i = 0; i < NCPU; ++i){
        initlock(&kmem[i].lock, kmem_lock_names[i]);
    }
    freerange(end, (void*) PHYSTOP);

}

void kfree(void *pa){
    r = (struct run*)pa;
    push_off();

    int cpu = cpuid();
    acquire(&kmem[cpu].lock);
    r->next = kmem[cpu].freelist;
    kmem[cpu].freelist = r;
    release(&kmem[cpu].lock);
    pop_off();
}

//根据题目，分配内存时候，可能会从其他CPU偷内存，共享数据修改，
//分配内存要加锁，偷页加锁
void  *kalloc(void){
    struct run *r;
    push_off();
    int cpu = cpuid();

    acquire(&kmem[cpu].lock);

    if (!kmem[cpu].freelist){
        int steal_left = 64;
        for (int i = 0; i < NCPU; ++i){
            if (i == cpu) continue;

            acquire(&kmem[i].lock);
            if (!kmem[i].freelist){
                release(&kmem[i].lock);
                continue;
            }

            struct run* rr = kmem[i].freelist;
            while (rr && steal_left){
                kmem[i].freelist = rr->next;
                rr->next = kmem[cpu].freelist;
                kmem[cpu].freelist = rr;
                rr = kmem[i].freelist;
                steal_left--;
            }
            release(&kmem[i].lock);
            if (steal_left == 0) break;
        }
    }
    r = kmem[cpu].freelist;
    if (r) kmem[cpu].freelist = r->next;
    release(&kmem[cpu].lock);

    pop_off();

    if (r) memset((char*) r, 5, PGSIZE);
    return (void *)r;
}

//one problem：死锁

//buffer cache
static struct *buf;
bget(uint dev, uint blockno){
    struct buf *b;
    acquire(&bcache.lock);

    for ( b = bcache.head.next; b != &bcache.head; b = b->next){
        if (b->dev == dev && b->blockno == blockno){
            b->refcnt++;
            release(&bcache.lock);
            acquiresleep(&b->lock);
            return b;
        }
    }

    for (b = bcache.head.prev; b != &bcache.head; b = b->prev){
        if (b->refcnt == 0){
            release(&bcache.lock);
            acquiresleep(&b->lock);
            return b;
        }
    }

    for (b = bcache.head.prev; b != &bcache.head; b = b->prev){
        if (b->refcnt == 0){
            b->dev = dev;
            b->blockno = blockno;
            b->valid = 0;
            b->refcnt = 1;
            release(&bcache.lock);
            acquiresleep(&b->lock);
            return b;
        }
    }
    //改进：建立一个由blockno和dev到buf的哈希表，通过一个特定的哈希公式映射到
    //哈希桶，由此在每个哈希桶上加锁。
}