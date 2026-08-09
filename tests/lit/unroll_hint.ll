; RUN: %opt %costforge -passes='costforge' -costforge-verbose -S < %s 2>&1 | %FileCheck %s
;
; Test: CostForge should set unroll hint on a simple reduction loop.
;
; CHECK: [CostForge]{{.*}}unroll
; CHECK: llvm.loop.unroll.count

define i32 @sum_array(ptr %arr, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %ptr = getelementptr i32, ptr %arr, i32 %i
  %val = load i32, ptr %ptr
  %sum.next = add i32 %sum, %val
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  %result = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  ret i32 %result
}
