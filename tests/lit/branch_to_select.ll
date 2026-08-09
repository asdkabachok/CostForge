; RUN: %opt %costforge -passes='costforge' -costforge-verbose -S < %s 2>&1 | %FileCheck %s
;
; Test: Simple diamond if/else with no side effects should be
;       converted to a select instruction (branchless).
;
; CHECK: select

define i32 @abs_val(i32 %x) {
entry:
  %cmp = icmp slt i32 %x, 0
  br i1 %cmp, label %neg, label %pos

neg:
  %negx = sub i32 0, %x
  br label %merge

pos:
  br label %merge

merge:
  %result = phi i32 [ %negx, %neg ], [ %x, %pos ]
  ret i32 %result
}
