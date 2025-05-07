//handle interupt + lazy allocation + multiprocessing

//trap机制
/*
每当执行程序系统调用，一个设备触发中断使得当前程序运行需要相应内核设备驱动
从只拥有user权限并且位于用户空间的程序，切换到拥有supervisor权限的内核，这个过程中，硬件
的状态会非常重要，因为很多的工作都是从硬件从适合运行用户应用程序的状态，到改变到适合运行内核代码的状态。

用户应用程序可以使用全部的32个寄存器，很多寄存器都有特殊的作用。用户应用程序可以使用
32个寄存器，很多寄存器都有特殊的作用，其中，程序技术寄存器PC，表明当前mode的标志位，表明是supervisor mode，还是
user mode，还有一堆控制CPU工作方式的寄存器，比如SATP，包含指向page table的物理内存地址。
STVEC，指向内核中处理trap指令的起始地址
寄存器表明了执行系统调用时，计算机的状态

在trap处理的过程中需要更改一些状态，或者对状态做一些操作。这样才可以运行系统内核中的程序。
先来预览一下需要做的操作。首先需要保存32个用户寄存器。

-shell调用write，作为用户代码调用write时，调用的是关联到shell的库函数，首先将SYS_write加载到
a7寄存器，之后这个函数中执行ecall，ecall会让程序计数器跳转到用户空间顶部trampoline page的位置，所以
现在指令正运行在内存的trampoline page中，这个page包含内核trap的处理代码。ecall并不会切换page table，
这是ecall指令的一个非常重要的特点。所以这意味着，trap处理代码必须存在在每一个user page table中，
因为ecall并不会切换page table，我们需要在user page table中的某个地方来执行最初的内核代码。
而这个trampoline page，是由内核小心映射到每一个user page table，使得我们仍在使用user page
table，内核在一个地方能够执行trap机制最开始的一些指令。

ecall三件事：
user mode -》supervisor mode
保存程序计数器的值
ecall会将STVEC拷贝到程序计数器，程序计数器执行STVEC所指向的代码，即trampoline page

cpy on write

线程具有状态，可以随时保存线程的状态并暂停线程的运行，通过恢复状态来恢复线程运行
线程三部分：PC，变量寄存器，程序stack
*/