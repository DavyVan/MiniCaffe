#include <cstdio>
#include "minicaffe/blob.h"

int main()
{
    Blob blob = Blob();
    blob.x = 2;
    blob.y = 3;
    blob.z = 4;
    printf("Hi MiniCaffe is running! %d\n", blob.get_ele_num());
}