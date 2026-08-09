; RUN: %opt %costforge -passes='costforge-ipa,costforge' -costforge-verbose -S < %s 2>&1 | %FileCheck %s
;
; Test: PGO branch_weights metadata should be read and used.
;       Function with entry_count should be classified.
;
; CHECK: [CostForge] PGO

define void @hot_loop(ptr %data, i64 %n) !prof !0 {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr i32, ptr %data, i64 %i
  %v = load i32, ptr %p
  %r = add i32 %v, 1
  store i32 %r, ptr %p
  %i.next = add i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop, !prof !1

exit:
  ret void
}

; Function entry count = 10000 (hot)
!0 = !{!"function_entry_count", i64 10000}
; Branch weights: loop back-edge taken 999900 times, exit 100 times
!1 = !{!"branch_weights", i32 100, i32 999900}
