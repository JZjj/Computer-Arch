//mit, 代码随想录

//sleep
int main(int argc, char **argv){
    if (argc < 2){
        prinf("usage: sleep <ticks>\n");
    }
    sleep(atoi(argv[1]));
    exit(0);
}

//pingpong
int main(int argc, char **argv){
    int pp2c[2], pc2p[2];
    pipe(pp2c);
    pipe(pc2p);

    if (fork() != 0){
        //父进程
        write(pp2c[1], ".", 1);
        close(pp2c[1]);

        //read from child process
        char buf;
        read(pc2p[0], &buf, 1);//read will block until write end closes
        printf("%d: received pong\n", getpid());
        wait(0);
    }
    else {
        char buf;
        read(pp2c[0], &buf, 1);
        printf("%d: received ping\n", getpid());

        write(pc3p[1], &buf, 1);
        close(pc2p[1]);
    }
    close(pp2c[0]);
    close(pc2p[0]);
    exit(0;)
}

//当前进程只会给右邻居写，不会从右邻居读，管道文件描述符，可以关闭读端或者写端
void sieve(int pleft[2]){
    int p;
    read(pleft[0], &p, sizeof(p));
    if (p == -1){
        exit(0;)
    }
    printf("prime %d\n", p);

    int pright[2];
    pipe(pright);

    if (fork() == 0){
        close(pright[1]);
        close(pleft[0]);
        sieve(pright);
    }
    else{
        close(pright[0]);
        int buf;
        while (read(pleft[0], &buf, sizeof(buf)) && buf != -1){
            if (buf % p != 0){
                write(pright[1], &buf, sizeof(buf));
            }
        }
        buf = -1;
        write(pright[1], &buf, sizeof(buf));
        wait(0);
        exit(0);
    }
}

int main (int argc, char **argv){
    int input_pipe[2];
    pipe(input_pipe);

    if (fork() == 0){
        close(input_pipe[1]);
        sieve(input_pipe);
        exit(0);
    }else{
        close(input_pipe[0]);
        int i;
        for (int i = 2; i <= 35; i++){
            write(input_pipe[1], &i, sizeof(i));
        }
        i = -1;
        write(input_pipe[1], &i, sizeof(i));
    }
    wait(0);
    exit(0);
}
