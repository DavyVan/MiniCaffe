# UCR-CS217-FinalProject

## How to build MiniCaffe

<code>
    cd [root_dir]/minicaffe 
    cmake -Bbuild .  
    cmake --build build  # Then find target library in /minicaffe/build/
</code>

## How to build with test

<code>
    cd [root_dir]/minicaffe
    cmake -Bbuild . -DTEST_ENABLED=ON
    cmake --build build     # Then find xxx_test in minicaffe/build/
</code>

## How to build main.cpp

<code>
    cd [root dir]
    cmake -Bbuild .
    cmake --build build  # Then find the final executable in [root dir]/build/
</code>