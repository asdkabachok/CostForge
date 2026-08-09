; RUN: %opt %costforge -passes='costforge' -costforge-verbose -S < %s 2>&1 | %FileCheck %s
;
; Test: CostForge should set vectorization hint on an independent
;       element-wise add loop.
;
; CHECK: [CostForge]{{.*}}vectorize
; CHECK: llvm.loop.vectorize.width

define void @vadd(ptr %a, ptr %b, ptr %c, i64 %n) {
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %bp = getelementptr float, ptr %b, i64 %i
  %cp = getelementptr float, ptr %c, i64 %i
  %ap = getelementptr float, ptr %a, i64 %i
  %bv = load float, ptr %bp
  %cv = load float, ptr %cp
  %sum = fadd float %bv, %cv
  store float %sum, ptr %ap
  %i.next = add i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
