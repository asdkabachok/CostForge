; RUN: %opt %costforge -passes='loop-simplify,costforge' -costforge-verbose -S < %s 2>&1 | %FileCheck %s
;
; Test: canonical i-j-k matmul reduction (C[i][j] = sum_k A[i][k]*B[k][j])
; over 2D array globals gets rewritten to i-k-j so every innermost-loop
; array access is unit-stride. The strided B[k][j]/C[i][j] access in the
; original j-k order is what made this pattern slow on real hardware
; (~2-4x, measured) regardless of any unroll/vectorize hint.
;
; CHECK: rewrote i-j-k matmul reduction as i-k-j

@A = global [384 x [384 x double]] zeroinitializer
@B = global [384 x [384 x double]] zeroinitializer
@C = global [384 x [384 x double]] zeroinitializer

define void @matmul() {
entry:
  br label %i.header

i.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %i.latch ]
  br label %j.header

j.header:
  %j = phi i64 [ 0, %i.header ], [ %j.next, %j.latch ]
  br label %k.header

k.header:
  %k = phi i64 [ 0, %j.header ], [ %k.next, %k.header ]
  %acc = phi double [ 0.000000e+00, %j.header ], [ %acc.next, %k.header ]
  %a.ptr = getelementptr inbounds [384 x [384 x double]], ptr @A, i64 0, i64 %i, i64 %k
  %a.val = load double, ptr %a.ptr
  %b.ptr = getelementptr inbounds [384 x [384 x double]], ptr @B, i64 0, i64 %k, i64 %j
  %b.val = load double, ptr %b.ptr
  %acc.next = call double @llvm.fmuladd.f64(double %a.val, double %b.val, double %acc)
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 384
  br i1 %k.done, label %j.latch, label %k.header

j.latch:
  %c.ptr = getelementptr inbounds [384 x [384 x double]], ptr @C, i64 0, i64 %i, i64 %j
  store double %acc.next, ptr %c.ptr
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 384
  br i1 %j.done, label %i.latch, label %j.header

i.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 384
  br i1 %i.done, label %exit, label %i.header

exit:
  ret void
}

declare double @llvm.fmuladd.f64(double, double, double)
