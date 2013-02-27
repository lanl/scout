// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Werror -Wno-address-of-temporary -U__declspec -D"__declspec(X)=" %t-modern-rw.cpp
// rdar:// 8243071
// REQUIRES: scoutdisable
// rdar://11375908
typedef unsigned long size_t;

void x(int y) {}
void f() {
    const int bar = 3;
    int baz = 4;
    __apple_block int bab = 4;
    __apple_block const int bas = 5;
    void (^b)() = ^{
        x(bar);
        x(baz);
        x(bab);
        x(bas);
        b();
    };
    b();
} 
