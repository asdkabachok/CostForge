; RUN: %opt %costforge -passes='costforge' -costforge-verbose -S < %s 2>&1 | %FileCheck %s
;
; Test: Loop latch branch should get 95:5 weight metadata.
; Test: Early-return error check should get 5:95 weight metadata.
;
; CHECK: !prof

define i32 @process(ptr %data, i32 %n) {
entry:
  %null = icmp eq ptr %data, null
  br i1 %null, label %error, label %loop.ph

error:
  ret i32 -1

loop.ph:
  br label %loop

loop:
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  %p = getelementptr i32, ptr %data, i32 %i
  %v = load i32, ptr %p
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret i32 0
}
