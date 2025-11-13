"""
Simulador de Pipeline In-Order de 5 Estágios com Cache Integrada
Arquitetura de Computadores II - Projeto 2
Subset RV32I com Cache L1I e L1D
"""

import csv
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

# ==================== CONFIGURAÇÕES ====================

class CacheConfig:
    """Configuração da Cache"""
    def __init__(self, size=1024, block_size=64, associativity=2, 
                 hit_latency=1, miss_penalty=100, policy='LRU', write_policy='WB'):
        self.size = size  # bytes
        self.block_size = block_size  # bytes
        self.associativity = associativity
        self.hit_latency = hit_latency
        self.miss_penalty = miss_penalty
        self.policy = policy  # LRU, FIFO, RAND
        self.write_policy = write_policy  # WB (Write-Back) ou WT (Write-Through)
        
        self.num_sets = size // (block_size * associativity)

# ==================== CACHE ====================

class CacheLine:
    """Linha de cache"""
    def __init__(self):
        self.valid = False
        self.dirty = False
        self.tag = 0
        self.data = bytearray(64)
        self.lru_counter = 0

class Cache:
    """Cache L1 (Instruções ou Dados)"""
    def __init__(self, config: CacheConfig, name: str):
        self.config = config
        self.name = name
        self.sets = [[CacheLine() for _ in range(config.associativity)] 
                     for _ in range(config.num_sets)]
        
        # Métricas
        self.hits = 0
        self.misses = 0
        self.accesses = 0
        
    def access(self, address: int, is_write: bool = False) -> int:
        """
        Acessa a cache. Retorna latência em ciclos.
        """
        self.accesses += 1
        
        block_offset_bits = (self.config.block_size - 1).bit_length()
        set_index_bits = (self.config.num_sets - 1).bit_length()
        
        set_index = (address >> block_offset_bits) & ((1 << set_index_bits) - 1)
        tag = address >> (block_offset_bits + set_index_bits)
        
        cache_set = self.sets[set_index]
        
        # Busca por hit
        for way in cache_set:
            if way.valid and way.tag == tag:
                self.hits += 1
                way.lru_counter = self.accesses  # Atualiza LRU
                if is_write and self.config.write_policy == 'WB':
                    way.dirty = True
                return self.config.hit_latency
        
        # Miss - precisa trazer da memória
        self.misses += 1
        latency = self.config.hit_latency + self.config.miss_penalty
        
        # Encontra linha para substituição
        victim = self._find_victim(cache_set)
        
        # Se write-back e linha suja, penalidade adicional
        if victim.valid and victim.dirty and self.config.write_policy == 'WB':
            latency += self.config.miss_penalty  # Write-back para memória
        
        # Carrega novo bloco
        victim.valid = True
        victim.tag = tag
        victim.dirty = is_write and self.config.write_policy == 'WB'
        victim.lru_counter = self.accesses
        
        return latency
    
    def _find_victim(self, cache_set: List[CacheLine]) -> CacheLine:
        """Encontra linha para substituição"""
        # Primeiro tenta achar linha inválida
        for way in cache_set:
            if not way.valid:
                return way
        
        # Política de substituição
        if self.config.policy == 'LRU':
            return min(cache_set, key=lambda x: x.lru_counter)
        elif self.config.policy == 'FIFO':
            return min(cache_set, key=lambda x: x.lru_counter)
        else:  # RAND
            import random
            return random.choice(cache_set)
    
    def get_mpki(self, instructions: int) -> float:
        """Calcula MPKI (Misses Per Kilo Instructions)"""
        if instructions == 0:
            return 0.0
        return (self.misses / instructions) * 1000
    
    def get_miss_rate(self) -> float:
        """Calcula taxa de miss"""
        if self.accesses == 0:
            return 0.0
        return self.misses / self.accesses
    
    def get_amat(self) -> float:
        """Calcula AMAT (Average Memory Access Time)"""
        miss_rate = self.get_miss_rate()
        return self.config.hit_latency + miss_rate * self.config.miss_penalty

# ==================== ISA ====================

class InstructionType(Enum):
    R_TYPE = 1  # add, sub, and, or, xor, slt
    I_TYPE = 2  # addi, lw, jalr
    S_TYPE = 3  # sw
    B_TYPE = 4  # beq, bne
    J_TYPE = 5  # jal

@dataclass
class Instruction:
    """Instrução RISC-V"""
    pc: int
    opcode: str
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0
    type: InstructionType = InstructionType.R_TYPE
    
    def __str__(self):
        if self.type == InstructionType.R_TYPE:
            return f"{self.opcode} x{self.rd}, x{self.rs1}, x{self.rs2}"
        elif self.type == InstructionType.I_TYPE:
            if self.opcode == 'lw':
                return f"{self.opcode} x{self.rd}, {self.imm}(x{self.rs1})"
            return f"{self.opcode} x{self.rd}, x{self.rs1}, {self.imm}"
        elif self.type == InstructionType.S_TYPE:
            return f"{self.opcode} x{self.rs2}, {self.imm}(x{self.rs1})"
        elif self.type == InstructionType.B_TYPE:
            return f"{self.opcode} x{self.rs1}, x{self.rs2}, {self.imm}"
        elif self.type == InstructionType.J_TYPE:
            return f"{self.opcode} x{self.rd}, {self.imm}"

# ==================== PREDITOR DE DESVIO ====================

class BranchPredictor:
    """Preditor de desvio"""
    def __init__(self, policy='2-bit'):
        self.policy = policy
        self.table = {}  # PC -> estado
        self.predictions = 0
        self.correct = 0
        
    def predict(self, pc: int) -> bool:
        """Retorna True se prediz desvio"""
        if self.policy == 'not-taken':
            return False
        elif self.policy == 'always-taken':
            return True
        elif self.policy == '1-bit':
            return self.table.get(pc, False)
        elif self.policy == '2-bit':
            # 0: Strongly Not Taken, 1: Weakly Not Taken
            # 2: Weakly Taken, 3: Strongly Taken
            state = self.table.get(pc, 1)
            return state >= 2
        return False
    
    def update(self, pc: int, taken: bool):
        """Atualiza preditor com resultado real"""
        self.predictions += 1
        predicted = self.predict(pc)
        
        if predicted == taken:
            self.correct += 1
        
        if self.policy == '1-bit':
            self.table[pc] = taken
        elif self.policy == '2-bit':
            state = self.table.get(pc, 1)
            if taken:
                self.table[pc] = min(3, state + 1)
            else:
                self.table[pc] = max(0, state - 1)
    
    def get_accuracy(self) -> float:
        """Retorna precisão do preditor"""
        if self.predictions == 0:
            return 0.0
        return self.correct / self.predictions

# ==================== PIPELINE ====================

@dataclass
class PipelineRegister:
    """Registrador de pipeline"""
    instruction: Optional[Instruction] = None
    pc: int = 0
    stalled: bool = False
    bubble: bool = True
    
    # EX
    alu_result: int = 0
    
    # MEM
    mem_data: int = 0
    mem_latency: int = 0
    
    # WB
    wb_data: int = 0

class Pipeline:
    """Pipeline de 5 estágios"""
    def __init__(self, l1i_config: CacheConfig, l1d_config: CacheConfig, 
                 predictor_policy='2-bit'):
        # Caches
        self.l1i = Cache(l1i_config, "L1I")
        self.l1d = Cache(l1d_config, "L1D")
        
        # Preditor
        self.predictor = BranchPredictor(predictor_policy)
        
        # Registradores de pipeline
        self.if_id = PipelineRegister()
        self.id_ex = PipelineRegister()
        self.ex_mem = PipelineRegister()
        self.mem_wb = PipelineRegister()
        
        # Registradores
        self.registers = [0] * 32
        self.registers[0] = 0  # x0 sempre zero
        
        # Memória de dados
        self.memory = bytearray(8192)  # 8KB de memória
        
        # Controle
        self.pc = 0
        self.cycle = 0
        self.instructions_committed = 0
        self.stalls = 0
        self.flushes = 0
        
        # Estado
        self.if_waiting = 0  # Ciclos esperando cache I
        self.mem_waiting = 0  # Ciclos esperando cache D
        self.program = []
        self.finished = False
        
    def load_program(self, instructions: List[Instruction]):
        """Carrega programa"""
        self.program = instructions
        self.pc = 0
        
    def run(self, max_cycles=10000):
        """Executa programa"""
        while not self.finished and self.cycle < max_cycles:
            self.cycle += 1
            
            # Pipeline stages (reverse order)
            self.stage_wb()
            self.stage_mem()
            self.stage_ex()
            self.stage_id()
            self.stage_if()
            
            # x0 sempre zero
            self.registers[0] = 0
            
        return self.get_metrics()
    
    def stage_if(self):
        """Instruction Fetch"""
        if self.if_waiting > 0:
            self.if_waiting -= 1
            return
        
        if self.if_id.stalled:
            return
        
        if self.pc >= len(self.program) * 4:
            self.finished = True
            return
        
        # Acessa cache de instruções
        latency = self.l1i.access(self.pc)
        
        if latency > 1:
            self.if_waiting = latency - 1
            return
        
        # Busca instrução
        inst_index = self.pc // 4
        if inst_index < len(self.program):
            inst = self.program[inst_index]
            self.if_id.instruction = inst
            self.if_id.pc = self.pc
            self.if_id.bubble = False
            
            # Predição para branches
            if inst.type == InstructionType.B_TYPE:
                if self.predictor.predict(self.pc):
                    self.pc = self.pc + inst.imm
                else:
                    self.pc += 4
            elif inst.type == InstructionType.J_TYPE:
                self.pc = self.pc + inst.imm
            else:
                self.pc += 4
        else:
            self.if_id.bubble = True
    
    def stage_id(self):
        """Instruction Decode"""
        if self.if_id.bubble or self.if_id.stalled:
            self.id_ex = PipelineRegister()
            return
        
        inst = self.if_id.instruction
        
        # Detecção de hazard
        if self._detect_hazard(inst):
            self.stalls += 1
            self.if_id.stalled = True
            self.id_ex = PipelineRegister()
            return
        
        self.if_id.stalled = False
        
        # Passa para EX
        self.id_ex.instruction = inst
        self.id_ex.pc = self.if_id.pc
        self.id_ex.bubble = False
        
        self.if_id = PipelineRegister()
    
    def stage_ex(self):
        """Execute"""
        if self.id_ex.bubble:
            self.ex_mem = PipelineRegister()
            return
        
        inst = self.id_ex.instruction
        
        # Lê registradores com forwarding
        rs1_val = self._get_forwarded_value(inst.rs1)
        rs2_val = self._get_forwarded_value(inst.rs2)
        
        # ALU
        result = 0
        if inst.opcode == 'add':
            result = rs1_val + rs2_val
        elif inst.opcode == 'sub':
            result = rs1_val - rs2_val
        elif inst.opcode == 'and':
            result = rs1_val & rs2_val
        elif inst.opcode == 'or':
            result = rs1_val | rs2_val
        elif inst.opcode == 'xor':
            result = rs1_val ^ rs2_val
        elif inst.opcode == 'slt':
            result = 1 if rs1_val < rs2_val else 0
        elif inst.opcode == 'addi':
            result = rs1_val + inst.imm
        elif inst.opcode in ['lw', 'sw']:
            result = rs1_val + inst.imm
        elif inst.opcode == 'jal':
            result = self.id_ex.pc + 4
        elif inst.opcode == 'jalr':
            result = self.id_ex.pc + 4
        
        # Verifica branch real
        if inst.type == InstructionType.B_TYPE:
            taken = False
            if inst.opcode == 'beq':
                taken = (rs1_val == rs2_val)
            elif inst.opcode == 'bne':
                taken = (rs1_val != rs2_val)
            
            predicted = self.predictor.predict(self.id_ex.pc)
            self.predictor.update(self.id_ex.pc, taken)
            
            if predicted != taken:
                # Misprediction - flush e corrige PC
                self.flushes += 1
                self.if_id = PipelineRegister()
                
                if taken:
                    self.pc = self.id_ex.pc + inst.imm
                else:
                    self.pc = self.id_ex.pc + 4
        
        self.ex_mem.instruction = inst
        self.ex_mem.pc = self.id_ex.pc
        self.ex_mem.alu_result = result
        self.ex_mem.bubble = False
        
        # Para SW, precisa do valor de rs2
        if inst.opcode == 'sw':
            self.ex_mem.mem_data = rs2_val
        
        self.id_ex = PipelineRegister()
    
    def stage_mem(self):
        """Memory Access"""
        if self.ex_mem.bubble:
            self.mem_wb = PipelineRegister()
            return
        
        if self.mem_waiting > 0:
            self.mem_waiting -= 1
            return
        
        inst = self.ex_mem.instruction
        result = self.ex_mem.alu_result
        
        if inst.opcode == 'lw':
            latency = self.l1d.access(result, is_write=False)
            if latency > 1:
                self.mem_waiting = latency - 1
                return
            
            # Lê da memória
            addr = result & 0x1FFF  # Máscara para 8KB
            value = int.from_bytes(self.memory[addr:addr+4], 'little')
            self.mem_wb.wb_data = value
            
        elif inst.opcode == 'sw':
            latency = self.l1d.access(result, is_write=True)
            if latency > 1:
                self.mem_waiting = latency - 1
                return
            
            # Escreve na memória
            addr = result & 0x1FFF
            self.memory[addr:addr+4] = self.ex_mem.mem_data.to_bytes(4, 'little', signed=True)
            
        else:
            self.mem_wb.wb_data = result
        
        self.mem_wb.instruction = inst
        self.mem_wb.pc = self.ex_mem.pc
        self.mem_wb.bubble = False
        
        self.ex_mem = PipelineRegister()
    
    def stage_wb(self):
        """Write Back"""
        if self.mem_wb.bubble:
            return
        
        inst = self.mem_wb.instruction
        
        if inst.rd != 0 and inst.opcode not in ['sw', 'beq', 'bne']:
            self.registers[inst.rd] = self.mem_wb.wb_data & 0xFFFFFFFF
        
        # Atualiza PC para JAL/JALR
        if inst.opcode == 'jalr':
            rs1_val = self.registers[inst.rs1]
            self.pc = (rs1_val + inst.imm) & ~1
        
        self.instructions_committed += 1
        self.mem_wb = PipelineRegister()
    
    def _detect_hazard(self, inst: Instruction) -> bool:
        """Detecta hazards de dados"""
        # Load-use hazard
        if self.id_ex.instruction and self.id_ex.instruction.opcode == 'lw':
            if (inst.rs1 == self.id_ex.instruction.rd or 
                inst.rs2 == self.id_ex.instruction.rd):
                return True
        return False
    
    def _get_forwarded_value(self, reg: int) -> int:
        """Obtém valor com forwarding"""
        if reg == 0:
            return 0
        
        # Forwarding de MEM
        if (self.ex_mem.instruction and 
            not self.ex_mem.bubble and
            self.ex_mem.instruction.rd == reg and 
            self.ex_mem.instruction.opcode != 'sw'):
            return self.ex_mem.alu_result
        
        # Forwarding de WB
        if (self.mem_wb.instruction and 
            not self.mem_wb.bubble and
            self.mem_wb.instruction.rd == reg):
            return self.mem_wb.wb_data
        
        return self.registers[reg]
    
    def get_metrics(self) -> Dict:
        """Retorna métricas"""
        cpi = self.cycle / self.instructions_committed if self.instructions_committed > 0 else 0
        ipc = 1 / cpi if cpi > 0 else 0
        
        return {
            'cycles': self.cycle,
            'instructions': self.instructions_committed,
            'cpi': cpi,
            'ipc': ipc,
            'stalls': self.stalls,
            'flushes': self.flushes,
            'branch_accuracy': self.predictor.get_accuracy(),
            'l1i_mpki': self.l1i.get_mpki(self.instructions_committed),
            'l1d_mpki': self.l1d.get_mpki(self.instructions_committed),
            'l1i_amat': self.l1i.get_amat(),
            'l1d_amat': self.l1d.get_amat(),
            'l1i_hits': self.l1i.hits,
            'l1i_misses': self.l1i.misses,
            'l1d_hits': self.l1d.hits,
            'l1d_misses': self.l1d.misses,
        }

# ==================== BENCHMARKS ====================

def create_benchmark_alu_sum(n=100):
    """Benchmark ALU-bound: somatório"""
    instructions = [
        # x1 = 0 (acumulador)
        Instruction(0, 'addi', rd=1, rs1=0, imm=0, type=InstructionType.I_TYPE),
        # x2 = n (contador)
        Instruction(4, 'addi', rd=2, rs1=0, imm=n, type=InstructionType.I_TYPE),
        # loop:
        # x1 = x1 + x2
        Instruction(8, 'add', rd=1, rs1=1, rs2=2, type=InstructionType.R_TYPE),
        # x2 = x2 - 1
        Instruction(12, 'addi', rd=2, rs1=2, imm=-1, type=InstructionType.I_TYPE),
        # if x2 != 0, goto loop
        Instruction(16, 'bne', rs1=2, rs2=0, imm=-12, type=InstructionType.B_TYPE),
    ]
    return instructions

def create_benchmark_mem_stride1(n=50):
    """Benchmark MEM-bound: acesso sequencial"""
    instructions = [
        # x1 = 0 (índice)
        Instruction(0, 'addi', rd=1, rs1=0, imm=0, type=InstructionType.I_TYPE),
        # x3 = n
        Instruction(4, 'addi', rd=3, rs1=0, imm=n, type=InstructionType.I_TYPE),
        # x4 = 1000 (base)
        Instruction(8, 'addi', rd=4, rs1=0, imm=1000, type=InstructionType.I_TYPE),
        # loop:
        # addr = x4 + x1
        Instruction(12, 'add', rd=5, rs1=4, rs2=1, type=InstructionType.R_TYPE),
        # lw x6, 0(x5)
        Instruction(16, 'lw', rd=6, rs1=5, imm=0, type=InstructionType.I_TYPE),
        # x1 = x1 + 4
        Instruction(20, 'addi', rd=1, rs1=1, imm=4, type=InstructionType.I_TYPE),
        # if x1 < x3, goto loop
        Instruction(24, 'bne', rs1=1, rs2=3, imm=-16, type=InstructionType.B_TYPE),
    ]
    return instructions

def create_benchmark_mem_stride8(n=50):
    """Benchmark MEM-bound: acesso com stride 8"""
    instructions = [
        Instruction(0, 'addi', rd=1, rs1=0, imm=0, type=InstructionType.I_TYPE),
        Instruction(4, 'addi', rd=3, rs1=0, imm=n, type=InstructionType.I_TYPE),
        Instruction(8, 'addi', rd=4, rs1=0, imm=1000, type=InstructionType.I_TYPE),
        Instruction(12, 'add', rd=5, rs1=4, rs2=1, type=InstructionType.R_TYPE),
        Instruction(16, 'lw', rd=6, rs1=5, imm=0, type=InstructionType.I_TYPE),
        Instruction(20, 'addi', rd=1, rs1=1, imm=32, type=InstructionType.I_TYPE),  # stride 8*4
        Instruction(24, 'bne', rs1=1, rs2=3, imm=-16, type=InstructionType.B_TYPE),
    ]
    return instructions

def create_benchmark_linear_search(n=30):
    """Benchmark controle: busca linear"""
    instructions = [
        Instruction(0, 'addi', rd=1, rs1=0, imm=0, type=InstructionType.I_TYPE),
        Instruction(4, 'addi', rd=2, rs1=0, imm=n, type=InstructionType.I_TYPE),
        Instruction(8, 'addi', rd=3, rs1=0, imm=25, type=InstructionType.I_TYPE),  # valor procurado
        Instruction(12, 'beq', rs1=1, rs2=3, imm=12, type=InstructionType.B_TYPE),  # achou
        Instruction(16, 'addi', rd=1, rs1=1, imm=1, type=InstructionType.I_TYPE),
        Instruction(20, 'bne', rs1=1, rs2=2, imm=-12, type=InstructionType.B_TYPE),
        # encontrado ou fim
        Instruction(24, 'addi', rd=4, rs1=0, imm=1, type=InstructionType.I_TYPE),
    ]
    return instructions

def create_benchmark_binary_search():
    """Benchmark controle: busca binária simplificada"""
    instructions = [
        Instruction(0, 'addi', rd=1, rs1=0, imm=0, type=InstructionType.I_TYPE),  # low
        Instruction(4, 'addi', rd=2, rs1=0, imm=31, type=InstructionType.I_TYPE),  # high
        Instruction(8, 'addi', rd=3, rs1=0, imm=15, type=InstructionType.I_TYPE),  # target
        # loop
        Instruction(12, 'add', rd=4, rs1=1, rs2=2, type=InstructionType.R_TYPE),  # mid = low+high
        Instruction(16, 'addi', rd=4, rs1=4, imm=-1, type=InstructionType.I_TYPE),  # mid/2 aprox
        Instruction(20, 'beq', rs1=4, rs2=3, imm=16, type=InstructionType.B_TYPE),  # found
        Instruction(24, 'slt', rd=5, rs1=4, rs2=3, type=InstructionType.R_TYPE),
        Instruction(28, 'bne', rs1=5, rs2=0, imm=8, type=InstructionType.B_TYPE),
        Instruction(32, 'addi', rd=2, rs1=4, imm=-1, type=InstructionType.I_TYPE),  # high=mid-1
        Instruction(36, 'bne', rs1=1, rs2=2, imm=-28, type=InstructionType.B_TYPE),
        Instruction(40, 'addi', rd=1, rs1=4, imm=1, type=InstructionType.I_TYPE),  # low=mid+1
        Instruction(44, 'bne', rs1=1, rs2=2, imm=-36, type=InstructionType.B_TYPE),
    ]
    return instructions

def create_benchmark_vector_reduction(n=64):
    """Benchmark ALU-bound: redução de vetor"""
    instructions = [
        Instruction(0, 'addi', rd=1, rs1=0, imm=0, type=InstructionType.I_TYPE),  # sum
        Instruction(4, 'addi', rd=2, rs1=0, imm=0, type=InstructionType.I_TYPE),  # index
        Instruction(8, 'addi', rd=3, rs1=0, imm=n, type=InstructionType.I_TYPE),  # limit
        # loop
        Instruction(12, 'addi', rd=1, rs1=1, imm=1, type=InstructionType.I_TYPE),  # sum += 1
        Instruction(16, 'addi', rd=2, rs1=2, imm=1, type=InstructionType.I_TYPE),  # index++
        Instruction(20, 'bne', rs1=2, rs2=3, imm=-12, type=InstructionType.B_TYPE),
    ]
    return instructions

# ==================== MAIN ====================

def run_experiments():
    """Executa experimentos e gera métricas"""
    
    benchmarks = {
        'alu_sum': create_benchmark_alu_sum(100),
        'alu_reduction': create_benchmark_vector_reduction(64),
        'mem_stride1': create_benchmark_mem_stride1(50),
        'mem_stride8': create_benchmark_mem_stride8(50),
        'ctrl_linear': create_benchmark_linear_search(30),
        'ctrl_binary': create_benchmark_binary_search(),
    }
    
    # Configurações de cache
    l1i_config = CacheConfig(size=1024, block_size=64, associativity=2, 
                             hit_latency=1, miss_penalty=100, policy='LRU')
    l1d_config = CacheConfig(size=1024, block_size=64, associativity=2, 
                             hit_latency=2, miss_penalty=100, policy='LRU', write_policy='WB')
    
    # Políticas de predição
    predictors = ['not-taken', '1-bit', '2-bit']
    
    results = []
    
    print("=" * 80)
    print("SIMULADOR DE PIPELINE COM CACHE INTEGRADA")
    print("=" * 80)
    
    for pred_policy in predictors:
        print(f"\n{'='*80}")
        print(f"Preditor: {pred_policy}")
        print(f"{'='*80}\n")
        
        for bench_name, program in benchmarks.items():
            # Cria pipeline
            pipeline = Pipeline(l1i_config, l1d_config, pred_policy)
            pipeline.load_program(program)
            
            # Executa
            metrics = pipeline.run(max_cycles=100000)
            
            # Armazena resultados
            result = {
                'benchmark': bench_name,
                'predictor': pred_policy,
                **metrics
            }
            results.append(result)
            
            # Imprime resumo
            print(f"\n{bench_name:20s} | CPI: {metrics['cpi']:.3f} | IPC: {metrics['ipc']:.3f}")
            print(f"  Ciclos: {metrics['cycles']:6d} | Instruções: {metrics['instructions']:6d}")
            print(f"  Stalls: {metrics['stalls']:6d} | Flushes: {metrics['flushes']:6d}")
            print(f"  Branch Acc: {metrics['branch_accuracy']:.2%}")
            print(f"  L1I MPKI: {metrics['l1i_mpki']:6.2f} | L1D MPKI: {metrics['l1d_mpki']:6.2f}")
            print(f"  L1I AMAT: {metrics['l1i_amat']:6.2f} | L1D AMAT: {metrics['l1d_amat']:6.2f}")
    
    # Salva resultados em CSV
    print(f"\n{'='*80}")
    print("Salvando métricas em 'resultados.csv'...")
    print(f"{'='*80}\n")
    
    with open('resultados.csv', 'w', newline='') as f:
        fieldnames = ['benchmark', 'predictor', 'cycles', 'instructions', 'cpi', 'ipc',
                     'stalls', 'flushes', 'branch_accuracy', 'l1i_mpki', 'l1d_mpki',
                     'l1i_amat', 'l1d_amat', 'l1i_hits', 'l1i_misses', 'l1d_hits', 'l1d_misses']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print("✓ Resultados salvos com sucesso!")
    
    # Análise comparativa
    print(f"\n{'='*80}")
    print("ANÁLISE COMPARATIVA")
    print(f"{'='*80}\n")
    
    # CPI médio por preditor
    print("CPI Médio por Preditor:")
    for pred in predictors:
        avg_cpi = sum(r['cpi'] for r in results if r['predictor'] == pred) / len(benchmarks)
        print(f"  {pred:15s}: {avg_cpi:.3f}")
    
    # Melhor preditor para cada benchmark
    print("\nMelhor Preditor por Benchmark (menor CPI):")
    for bench_name in benchmarks.keys():
        bench_results = [r for r in results if r['benchmark'] == bench_name]
        best = min(bench_results, key=lambda x: x['cpi'])
        print(f"  {bench_name:20s}: {best['predictor']:15s} (CPI: {best['cpi']:.3f})")
    
    # Características dos benchmarks
    print("\nCaracterísticas dos Benchmarks:")
    print(f"  {'Benchmark':<20s} {'Tipo':<12s} {'Stalls':<8s} {'Flushes':<8s} {'L1D MPKI':<10s}")
    print(f"  {'-'*70}")
    
    for bench_name in benchmarks.keys():
        # Pega resultado com preditor 2-bit
        result = next(r for r in results if r['benchmark'] == bench_name and r['predictor'] == '2-bit')
        
        if 'alu' in bench_name:
            tipo = 'ALU-bound'
        elif 'mem' in bench_name:
            tipo = 'MEM-bound'
        else:
            tipo = 'CTRL-bound'
        
        print(f"  {bench_name:<20s} {tipo:<12s} {result['stalls']:<8d} "
              f"{result['flushes']:<8d} {result['l1d_mpki']:<10.2f}")
    
    return results

def run_cache_sensitivity():
    """Experimento de sensibilidade ao tamanho da cache"""
    print(f"\n{'='*80}")
    print("EXPERIMENTO: SENSIBILIDADE AO TAMANHO DA CACHE")
    print(f"{'='*80}\n")
    
    benchmark = create_benchmark_mem_stride1(100)
    cache_sizes = [512, 1024, 2048, 4096]
    
    results = []
    
    for size in cache_sizes:
        l1i_config = CacheConfig(size=1024, block_size=64, associativity=2, 
                                 hit_latency=1, miss_penalty=100)
        l1d_config = CacheConfig(size=size, block_size=64, associativity=2, 
                                 hit_latency=2, miss_penalty=100, policy='LRU', write_policy='WB')
        
        pipeline = Pipeline(l1i_config, l1d_config, '2-bit')
        pipeline.load_program(benchmark)
        metrics = pipeline.run()
        
        results.append({
            'cache_size': size,
            'cpi': metrics['cpi'],
            'l1d_mpki': metrics['l1d_mpki'],
            'l1d_amat': metrics['l1d_amat']
        })
        
        print(f"L1D: {size:4d}B | CPI: {metrics['cpi']:.3f} | "
              f"MPKI: {metrics['l1d_mpki']:.2f} | AMAT: {metrics['l1d_amat']:.2f}")
    
    # Salva em CSV
    with open('cache_sensitivity.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cache_size', 'cpi', 'l1d_mpki', 'l1d_amat'])
        writer.writeheader()
        writer.writerows(results)
    
    print("\n✓ Resultados salvos em 'cache_sensitivity.csv'")

def run_write_policy_comparison():
    """Compara políticas de escrita"""
    print(f"\n{'='*80}")
    print("EXPERIMENTO: COMPARAÇÃO DE POLÍTICAS DE ESCRITA")
    print(f"{'='*80}\n")
    
    # Benchmark com muitas escritas
    benchmark = [
        Instruction(0, 'addi', rd=1, rs1=0, imm=0, type=InstructionType.I_TYPE),
        Instruction(4, 'addi', rd=2, rs1=0, imm=100, type=InstructionType.I_TYPE),
        Instruction(8, 'addi', rd=3, rs1=0, imm=1000, type=InstructionType.I_TYPE),
        # loop: escreve na memória
        Instruction(12, 'add', rd=4, rs1=3, rs2=1, type=InstructionType.R_TYPE),
        Instruction(16, 'sw', rs1=4, rs2=1, imm=0, type=InstructionType.S_TYPE),
        Instruction(20, 'addi', rd=1, rs1=1, imm=4, type=InstructionType.I_TYPE),
        Instruction(24, 'bne', rs1=1, rs2=2, imm=-16, type=InstructionType.B_TYPE),
    ]
    
    policies = ['WB', 'WT']
    results = []
    
    for policy in policies:
        l1i_config = CacheConfig(size=1024, block_size=64, associativity=2, 
                                 hit_latency=1, miss_penalty=100)
        l1d_config = CacheConfig(size=1024, block_size=64, associativity=2, 
                                 hit_latency=2, miss_penalty=100, 
                                 policy='LRU', write_policy=policy)
        
        pipeline = Pipeline(l1i_config, l1d_config, '2-bit')
        pipeline.load_program(benchmark)
        metrics = pipeline.run()
        
        results.append({
            'policy': policy,
            'cpi': metrics['cpi'],
            'cycles': metrics['cycles']
        })
        
        print(f"{policy:2s} | CPI: {metrics['cpi']:.3f} | Ciclos: {metrics['cycles']:6d}")
    
    print(f"\nObservação:")
    print(f"  - Write-Back (WB): Melhor desempenho, escreve na memória apenas no evict")
    print(f"  - Write-Through (WT): Todas as escritas vão para memória (mais lento)")

def demo_pipeline_trace():
    """Demonstração com trace detalhado do pipeline"""
    print(f"\n{'='*80}")
    print("DEMONSTRAÇÃO: TRACE DO PIPELINE")
    print(f"{'='*80}\n")
    
    # Programa simples
    program = [
        Instruction(0, 'addi', rd=1, rs1=0, imm=5, type=InstructionType.I_TYPE),   # x1 = 5
        Instruction(4, 'addi', rd=2, rs1=0, imm=3, type=InstructionType.I_TYPE),   # x2 = 3
        Instruction(8, 'add', rd=3, rs1=1, rs2=2, type=InstructionType.R_TYPE),    # x3 = x1 + x2
        Instruction(12, 'sub', rd=4, rs1=3, rs2=2, type=InstructionType.R_TYPE),   # x4 = x3 - x2
        Instruction(16, 'addi', rd=5, rs1=4, imm=10, type=InstructionType.I_TYPE), # x5 = x4 + 10
    ]
    
    print("Programa:")
    for i, inst in enumerate(program):
        print(f"  [{i*4:3d}] {inst}")
    
    l1i_config = CacheConfig(size=1024, block_size=64, associativity=2, 
                             hit_latency=1, miss_penalty=10)
    l1d_config = CacheConfig(size=1024, block_size=64, associativity=2, 
                             hit_latency=1, miss_penalty=10)
    
    pipeline = Pipeline(l1i_config, l1d_config, 'not-taken')
    pipeline.load_program(program)
    metrics = pipeline.run()
    
    print(f"\nResultados:")
    print(f"  Ciclos totais: {metrics['cycles']}")
    print(f"  Instruções: {metrics['instructions']}")
    print(f"  CPI: {metrics['cpi']:.3f}")
    print(f"  IPC: {metrics['ipc']:.3f}")
    print(f"  Stalls: {metrics['stalls']}")
    print(f"  Flushes: {metrics['flushes']}")
    
    print(f"\nRegistradores finais:")
    for i in range(1, 6):
        print(f"  x{i} = {pipeline.registers[i]}")

def print_config_info():
    """Imprime informações sobre as configurações"""
    print("\n" + "="*80)
    print("CONFIGURAÇÕES DO SIMULADOR")
    print("="*80)
    
    print("\nArquitetura:")
    print("  - ISA: RV32I (subset)")
    print("  - Pipeline: 5 estágios (IF, ID, EX, MEM, WB)")
    print("  - Forwarding: EX/MEM → ID/EX, MEM/WB → ID/EX")
    print("  - Hazards: Detecção com stalls quando necessário")
    
    print("\nCache L1I (Instruções):")
    print("  - Tamanho: 1 KB")
    print("  - Bloco: 64 bytes")
    print("  - Associatividade: 2 vias")
    print("  - Hit latency: 1 ciclo")
    print("  - Miss penalty: 100 ciclos")
    print("  - Política: LRU")
    
    print("\nCache L1D (Dados):")
    print("  - Tamanho: 1 KB")
    print("  - Bloco: 64 bytes")
    print("  - Associatividade: 2 vias")
    print("  - Hit latency: 2 ciclos")
    print("  - Miss penalty: 100 ciclos")
    print("  - Política: LRU")
    print("  - Escrita: Write-Back com Write-Allocate")
    
    print("\nPreditores de Desvio:")
    print("  - not-taken: Sempre prediz não desviar")
    print("  - 1-bit: Saturating counter de 1 bit")
    print("  - 2-bit: Saturating counter de 2 bits (bimodal)")
    
    print("\nBenchmarks:")
    print("  - alu_sum: Somatório (ALU-bound)")
    print("  - alu_reduction: Redução de vetor (ALU-bound)")
    print("  - mem_stride1: Acesso sequencial (MEM-bound)")
    print("  - mem_stride8: Acesso com stride 8 (MEM-bound)")
    print("  - ctrl_linear: Busca linear (CTRL-bound)")
    print("  - ctrl_binary: Busca binária (CTRL-bound)")

if __name__ == "__main__":
    print_config_info()
    
    # Executa experimentos principais
    results = run_experiments()
    
    # Experimentos adicionais
    run_cache_sensitivity()
    run_write_policy_comparison()
    demo_pipeline_trace()
    
    print(f"\n{'='*80}")
    print("SIMULAÇÃO CONCLUÍDA!")
    print(f"{'='*80}")
    print("\nArquivos gerados:")
    print("  - resultados.csv: Métricas completas de todos os experimentos")
    print("  - cache_sensitivity.csv: Análise de sensibilidade ao tamanho da cache")
    print("\nPara visualizar os resultados, você pode:")
    print("  1. Abrir os arquivos CSV em uma planilha")
    print("  2. Usar pandas/matplotlib para gerar gráficos")
    print("  3. Analisar o impacto de diferentes configurações no desempenho")
    print("="*80 + "\n")
