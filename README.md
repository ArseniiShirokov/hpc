### Описание файлов
main.c - содержит решение волнового уравнения методом конечных разностей

### Компиляция и запуск
```console
gcc -fopenmp main.c -o run -lm
./run L_x L_y L_z T N K num_threads
```
