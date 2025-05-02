//file system study note
//学习资料：mit，代码随想录
/*
操作系统中，系统调用是程序和内核交互的关键。
用户程序通过调用库函数间接发起系统调用，系统调用触发软件中断或陷阱，引发CPU切换到内核态，
内核通过系统调用号确定要执行的服务，完成服务后，将结果或错误码存入用户程序的寄存器，并返回到用户态
继续执行程序。

1. 用户执行read系统调用
2. 执行read函数
3. kernel/trampoline.S assembly code transition between CPU mode
4. kernel/trap.c
5. kernel/syscall.c
6. sys_read
7. kernel/trap.c usertrapret
8. kernel/trampoline.S userret


//system call tracing
首先加入函数声明
int trace(int);

在进程类proc结构中，添加kama_syscall_trace
*/
uint64 sys_trace(void){
    int mask;

    if (argint(0, &mask) < 0) return -1;
    myproc()->kama_syscall_trace = mask;
    return 0;
}

void syscall(){
    int num;
    struct proc *p = myproc();

    num = p->trapframe->a7;
    if (num > 0 && num < NELEM(syscalls) && syscalls[num]){
        //如果系统调用编号有效，调用对应的处理函数，并返回值存储在a0寄存器中
        p->trapframe->a0 = syscalls[num]();

        //如果当前进程启用了trace跟踪，则按照题目设计要求打印信息
        if ((p->kama_syscall_trace >> num) & 1){
            printf("%d: syscall %s -> %d\n", p->pid, kama_syscall_names[num], p->trapframe->a0);
        }
    }
    else{
        printf("%d %s: unknown sys call %d\n", p->pid, p->name, num);
        p->trapframe->a0 = -1;
    }
}
