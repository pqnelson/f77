	.file	"do1.f"
# GNU Fortran2008 (Ubuntu 11.4.0-1ubuntu1~22.04) version 11.4.0 (x86_64-linux-gnu)
#	compiled by GNU C version 11.4.0, GMP version 6.2.1, MPFR version 4.1.0, MPC version 1.2.1, isl version isl-0.24-GMP

# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed: -ffixed-form -mtune=generic -march=x86-64 -O0 -std=legacy -fintrinsic-modules-path /usr/lib/gcc/x86_64-linux-gnu/11/finclude -fpre-include=/usr/include/finclude/math-vector-fortran.h
	.text
	.section	.rodata
.LC0:
	.string	"do1.f"
.LC1:
	.ascii	"Done with loop, goodbye"
	.text
	.type	MAIN__, @function
MAIN__:
.LFB0:
	.cfi_startproc
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	subq	$544, %rsp	#,
# do1.f:9:       DO 10 i=1, 0
	movl	$1, -4(%rbp)	#, i
.L3:
# do1.f:9:       DO 10 i=1, 0
	movl	-4(%rbp), %eax	# i, i.3_1
	testl	%eax, %eax	# i.3_1
	setg	%al	#, tmp86
	movzbl	%al, %eax	# tmp86, _7
	testl	%eax, %eax	# _7
	jne	.L4	#,
# do1.f:10:          WRITE (*,*) i
	leaq	.LC0(%rip), %rax	#, tmp87
	movq	%rax, -536(%rbp)	# tmp87, MEM[(struct __st_parameter_dt *)_26].common.filename
	movl	$10, -528(%rbp)	#, MEM[(struct __st_parameter_dt *)_26].common.line
	movl	$128, -544(%rbp)	#, MEM[(struct __st_parameter_dt *)_26].common.flags
	movl	$6, -540(%rbp)	#, MEM[(struct __st_parameter_dt *)_26].common.unit
	leaq	-544(%rbp), %rax	#, tmp88
	movq	%rax, %rdi	# tmp88,
	call	_gfortran_st_write@PLT	#
	leaq	-4(%rbp), %rcx	#, tmp89
	leaq	-544(%rbp), %rax	#, tmp90
	movl	$4, %edx	#,
	movq	%rcx, %rsi	# tmp89,
	movq	%rax, %rdi	# tmp90,
	call	_gfortran_transfer_integer_write@PLT	#
	leaq	-544(%rbp), %rax	#, tmp91
	movq	%rax, %rdi	# tmp91,
	call	_gfortran_st_write_done@PLT	#
# do1.f:9:       DO 10 i=1, 0
	movl	-4(%rbp), %eax	# i, i.4_2
	addl	$1, %eax	#, _3
	movl	%eax, -4(%rbp)	# _3, i
	jmp	.L3	#
.L4:
# do1.f:11:  10   CONTINUE
	nop	
# do1.f:13:       WRITE (*,*) 'Done with loop, goodbye'
	leaq	.LC0(%rip), %rax	#, tmp92
	movq	%rax, -536(%rbp)	# tmp92, MEM[(struct __st_parameter_dt *)_26].common.filename
	movl	$13, -528(%rbp)	#, MEM[(struct __st_parameter_dt *)_26].common.line
	movl	$128, -544(%rbp)	#, MEM[(struct __st_parameter_dt *)_26].common.flags
	movl	$6, -540(%rbp)	#, MEM[(struct __st_parameter_dt *)_26].common.unit
	leaq	-544(%rbp), %rax	#, tmp93
	movq	%rax, %rdi	# tmp93,
	call	_gfortran_st_write@PLT	#
# do1.f:13:       WRITE (*,*) 'Done with loop, goodbye'
	leaq	-544(%rbp), %rax	#, tmp94
	movl	$23, %edx	#,
	leaq	.LC1(%rip), %rcx	#, tmp95
	movq	%rcx, %rsi	# tmp95,
	movq	%rax, %rdi	# tmp94,
	call	_gfortran_transfer_character_write@PLT	#
# do1.f:13:       WRITE (*,*) 'Done with loop, goodbye'
	leaq	-544(%rbp), %rax	#, tmp96
	movq	%rax, %rdi	# tmp96,
	call	_gfortran_st_write_done@PLT	#
# do1.f:15:       STOP
	movl	$0, %edx	#,
	movl	$0, %esi	#,
	movl	$0, %edi	#,
	call	_gfortran_stop_string@PLT	#
	.cfi_endproc
.LFE0:
	.size	MAIN__, .-MAIN__
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	subq	$16, %rsp	#,
	movl	%edi, -4(%rbp)	# argc, argc
	movq	%rsi, -16(%rbp)	# argv, argv
# do1.f:16:       END
	movq	-16(%rbp), %rdx	# argv, tmp84
	movl	-4(%rbp), %eax	# argc, tmp85
	movq	%rdx, %rsi	# tmp84,
	movl	%eax, %edi	# tmp85,
	call	_gfortran_set_args@PLT	#
	leaq	options.2.0(%rip), %rax	#, tmp86
	movq	%rax, %rsi	# tmp86,
	movl	$7, %edi	#,
	call	_gfortran_set_options@PLT	#
	call	MAIN__	#
	movl	$0, %eax	#, _7
	leave	
	.cfi_def_cfa 7, 8
	ret	
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
	.align 16
	.type	options.2.0, @object
	.size	options.2.0, 28
options.2.0:
	.long	0
	.long	4095
	.long	0
	.long	1
	.long	1
	.long	0
	.long	31
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
