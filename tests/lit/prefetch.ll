; RUN: %opt %costforge -passes='costforge' -costforge-verbose -S < %s 2>&1 | %FileCheck %s
;
; Test: Streaming load in a loop with GEP should get prefetch inserted.
;
; CHECK: llvm.prefetch

define double @dot_product(ptr %a, ptr %b, i64 %n) {
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi double [ 0.0, %entry ], [ %sum.next, %loop ]
  %ap = getelementptr double, ptr %a, i64 %i
  %bp = getelementptr double, ptr %b, i64 %i
  %av = load double, ptr %ap
  %bv = load double, ptr %bp
  %prod = fmul double %av, %bv
  %sum.next = fadd double %sum, %prod
  %i.next = add i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  %result = phi double [ 0.0, %entry ], [ %sum.next, %loop ]
  ret double %result
}
