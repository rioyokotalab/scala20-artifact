#pragma once

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

extern const int64_t B_SIZE;
extern const int64_t VEC_SIZE;
extern const int64_t extra;

extern int64_t B_small;
extern int64_t B_skinny;
extern int64_t K_SIZE;
extern double packA_time, packB_time, mk_time;
extern int64_t batch_size, block_size, rank;
extern int64_t MB, NB, XB, YB, LDA_V, LDB_U, LDB_S, LDA_S;
extern double packA_V_time, packB_S_time, packB_U_time, packA_S_time;

#define GEMM_SIMD_ALIGN_SIZE 64

typedef unsigned long long ticks;


ticks getticks(void);
void allocate_data(double ** &A, double ** &B, double ** &C,
                   double ** &Asim, double ** &Bsim, double ** &Csim,
                   int M, int N, int K);
void allocate_data(double ** &A, double ** &Asim, int M, int N, double value=-1);
void free_data(double ** A, double ** Asim);
void free_data(double ** A, double ** B, double ** C, double ** Asim, double ** Bsim, double ** Csim);
void print (double * mat, int M , int N, int LDmat);
double *malloc_aligned(int m, int n);
double *malloc_aligned(int size);

class Timer
{
public:
  inline void start()
  {
    m_bRunning = true;
    m_StartTime = std::chrono::system_clock::now();
  }
    
  inline void stop()
  {
    m_EndTime = std::chrono::system_clock::now();
    m_bRunning = false;
  }
    
  double time_ms()
  {
    std::chrono::time_point<std::chrono::system_clock> endTime;
        
    if(m_bRunning)
      {
        endTime = std::chrono::system_clock::now();
      }
    else
      {
        endTime = m_EndTime;
      }
        
    return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_StartTime).count();
  }
    
  double time_s()
  {
    return time_ms() / 1000.0;
  }

  double time_ns() {
    std::chrono::time_point<std::chrono::system_clock> endTime;
        
    if(m_bRunning) {
        endTime = std::chrono::system_clock::now();
    }
    else {
        endTime = m_EndTime;
    }
        
    return std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - m_StartTime).count();    
  }

private:
  std::chrono::time_point<std::chrono::system_clock> m_StartTime;
  std::chrono::time_point<std::chrono::system_clock> m_EndTime;
  bool                                               m_bRunning = false;
};

extern Timer timer;

// At&T style assembly blocks.

// Blocks for construction of commands.
#define COMMENT_BEGIN "#"
#define COMMENT_END

#define BEGIN_ASM() __asm__ volatile (
#define END_ASM(...) __VA_ARGS__ );

#define STRINGIFY_(...) #__VA_ARGS__
#define GET_MACRO_(_1_,_2_,_3_,_4_,NAME,...) NAME

// Blocks for construction of instructions.

#define LABEL_(label) ".L" STRINGIFY_(label) "%="
#define INSTR_(name,...) GET_MACRO_(__VA_ARGS__,INSTR_4_,INSTR_3_,INSTR_2_, \
                                    INSTR_1_,INSTR_0_)(name,__VA_ARGS__)

#define INSTR_4_(name,_0,_1,_2,_3) STRINGIFY_(name) " " STRINGIFY_(_0,_1,_2,_3) "\n\t"
#define INSTR_3_(name,_0,_1,_2) STRINGIFY_(name) " " STRINGIFY_(_0,_1,_2) "\n\t"
#define INSTR_2_(name,_0,_1) STRINGIFY_(name) " " STRINGIFY_(_0,_1) "\n\t"
#define INSTR_1_(name,_0) STRINGIFY_(name) " " STRINGIFY_(_0) "\n\t"
#define INSTR_0_(name) STRINGIFY_(name) "\n\t"

#define LABEL(target) LABEL_(target) ":\n\t"
#define REGISTER_(r) r
#define IMM(x) x
#define VAR(x) %[x]
#define MASK_(x) %{x%}
#define JMP_(insn, target) STRINGIFY_(insn) " " LABEL_(target) "\n\t"

#define MEM_4_(reg,off,scale,disp) [reg + off*scale + disp]
#define MEM_3_(reg,off,scale) [reg + off*scale]
#define MEM_2_(reg,disp) [reg + disp]
#define MEM_1_(reg) [reg]

// General-purpose registers

#define AL REGISTER_(al)
#define AH REGISTER_(ah)
#define BL REGISTER_(bl)
#define BH REGISTER_(bh)
#define CL REGISTER_(cl)
#define CH REGISTER_(ch)
#define DL REGISTER_(dl)
#define DH REGISTER_(dh)
#define R8B REGISTER_(r8b)
#define R9B REGISTER_(r9b)
#define R10B REGISTER_(r10b)
#define R11B REGISTER_(r11b)
#define R12B REGISTER_(r12b)
#define R13B REGISTER_(r13b)
#define R14B REGISTER_(r14b)
#define R15B REGISTER_(r15b)

#define EAX REGISTER_(eax)
#define EBX REGISTER_(ebx)
#define ECX REGISTER_(ecx)
#define EDX REGISTER_(edx)
#define ESP REGISTER_(esp)
#define EBP REGISTER_(ebp)
#define EDI REGISTER_(edi)
#define ESI REGISTER_(esi)
#define R8D REGISTER_(r8d)
#define R9D REGISTER_(r9d)
#define R10D REGISTER_(r10d)
#define R11D REGISTER_(r11d)
#define R12D REGISTER_(r12d)
#define R13D REGISTER_(r13d)
#define R14D REGISTER_(r14d)
#define R15D REGISTER_(r15d)

#define RAX REGISTER_(rax)
#define RBX REGISTER_(rbx)
#define RCX REGISTER_(rcx)
#define RDX REGISTER_(rdx)
#define RSP REGISTER_(rsp)
#define RBP REGISTER_(rbp)
#define RDI REGISTER_(rdi)
#define RSI REGISTER_(rsi)
#define R8 REGISTER_(r8)
#define R9 REGISTER_(r9)
#define R10 REGISTER_(r10)
#define R11 REGISTER_(r11)
#define R12 REGISTER_(r12)
#define R13 REGISTER_(r13)
#define R14 REGISTER_(r14)
#define R15 REGISTER_(r15)

// Vector registers

#define XMM(x) REGISTER_(Xmm##x)
#define YMM(x) REGISTER_(Ymm##x)
#define ZMM(x) REGISTER_(Zmm##x)
#define K(x) REGISTER_(k##x)
#define MASK_K(n) MASK_(K(n))
#define MASK_KZ(n) MASK_(K(n))MASK_(z)

#define MEM(...) GET_MACRO_(__VA_ARGS__,MEM_4_,MEM_3_,MEM_2_,MEM_1_)(__VA_ARGS__)

// Instructions specify the intel syntax by defualt.
  
// Bitwise instructions

#define VXORPD(_0, _1, _2) INSTR_(vxorpd, _0, _1, _2)
#define AND(_0, _1) INSTR_(and, _0, _1)
#define SAR(...) INSTR_(sar, __VA_ARGS__)

// Integer arithmetic
  
#define ADD(_0, _1) INSTR_(add, _0, _1)
#define SUB(_0, _1) INSTR_(sub, _0, _1)
#define DEC(_0) INSTR_(dec, _0)
#define INC(_0) INSTR_(inc, _0)

// Arithmetic

#define VFMADD231PD(_0, _1, _2) INSTR_(vfmadd231pd, _0, _1, _2)

// Memory access

#define LEA(_0, _1) INSTR_(lea, _0, _1)
#define MOV(_0, _1) INSTR_(mov, _0, _1)
#define MOVD(_0, _1) INSTR_(movd, _0, _1)
#define MOVL(_0, _1) INSTR_(movl, _0, _1)
#define MOVQ(_0, _1) INSTR_(movq, _0, _1)

#define VPMULLQ(_0, _1, _2) INSTR_(vpmullq, _0, _1, _2)
#define VMULPD(_0, _1, _2) INSTR_(vmulpd, _0, _1, _2)
#define VMOVAPD(_0, _1) INSTR_(vmovapd, _0, _1)
#define VBROADCASTSD(_0, _1) INSTR_(vbroadcastsd, _0, _1)
#define VPBROADCASTD(_0, _1) INSTR_(vpbroadcastd, _0, _1)
#define VPBROADCASTQ(_0, _1) INSTR_(vpbroadcastq, _0, _1)
#define VMOVUPD(_0, _1) INSTR_(vmovupd, _0, _1) // move unaligned packed data from/to ZMM register.
#define VGATHERQPD(...) INSTR_(vgatherqpd, __VA_ARGS__)
#define VSCATTERQPD(_0, _1) INSTR_(vscatterqpd, _0, _1)

// Masking operations

#define KXNORW(_0, _1, _2) INSTR_(kxnorw, _0, _1, _2)

// jump access

#define JC(_0) JMP_(jc, _0)
#define JB(_0) JC(_0)
#define JNAE(_0) JC(_0)
#define JNC(_0) JMP_(jnc, _0)
#define JNB(_0) JNC(_0)
#define JAE(_0) JNC(_0)

#define JG(_0) JMP_(jg, _0)
#define JNLE(_0) JG(_0)
#define JNG(_0) JMP_(jng, _0)
#define JLE(_0) JNG(_0)
#define JNZ(_0) JNE(_0)
#define JE(_0) JMP_(je, _0)
#define JZ(_0) JE(_0)
#define JNE(_0) JMP_(jne, _0)

// COmparisons


#define CMP(_0, _1) INSTR_(cmp, _0, _1)
#define TEST(_0, _1) INSTR_(test, _0, _1)

// prefetches
#define PREFETCH(_0, _1) INSTR_(prefetcht##_0, _1)
#define PREFETCHW0(_0) INSTR_(prefetchw, _0)
#define PREFETCHW1(_0) INSTR_(prefetchwt1, _0)
#define VGATHERPFDPS(_0, _1) INSTR_(vgatherpf##_0##dps, _1)
#define VSCATTERPFDPS(_0, _1) INSTR_(vscatterpf##_0##dps, _1)
#define VGATHERPFDPD(_0, _1) INSTR_(vgatherpf##_0##dpd, _1)
#define VSCATTERPFDPD(_0, _1) INSTR_(vscatterpf##_0##dpd, _1)
#define VGATHERPFQPS(_0, _1) INSTR_(vgatherpf##_0##qps, _1)
#define VSCATTERPFQPS(_0, _1) INSTR_(vscatterpf##_0##qps, _1)
#define VGATHERPFQPD(_0, _1) INSTR_(vgatherpf##_0##qpd, _1)
#define VSCATTERPFQPD(_0, _1) INSTR_(vscatterpf##_0##qpd, _1)
