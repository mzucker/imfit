imfit
======
Image fitting with Gabor functions

Building
========

To build the software, you will need these libraries:
 
  - opencv <http://opencv.org/>
  - levmar <http://users.ics.forth.gr/~lourakis/levmar/>
  
You must install cmake to build the software as well.  To build:

    cd /path/to/imfit
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make

Using the software
==================

Try running 

    ./imfit -s128 -n128 -i ../params/zz_rect.txt -w ../images/zz_rect_weights.png ../images/zz_rect.png 

See also
========

  - <https://www.shadertoy.com/view/4ljSRR>
  - <https://www.shadertoy.com/view/XltGzS>
