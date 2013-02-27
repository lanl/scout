// RUN: %clang_cc1 -x objective-c++ -fblocks -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 -o - %s
// REQUIRES: scoutdisable

@interface Foo {
    void (^_block)(void);
}
@end

@implementation Foo
- (void)bar {
    _block();
}
@end
