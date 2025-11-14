from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum

# ============================================================================
# 1. NÚCLEO DO PIPELINE
# ============================================================================

@dataclass
class PipelineStage:
    """Registrador entre estágios do pipeline"""
    instruction: Optional[int] = None
    pc: int = 0
    rs1_val: int = 0
    rs2_val: int = 0
    alu_result: int = 0
    mem_data: int = 0
    rd: int = 0
    is_bubble: bool = True

class Pipeline:
    def __init__(self):
        self.pc = 0
        self.registers = [0] * 32  # Banco de registradores
        self.if_id = PipelineStage()
        self.id_ex = PipelineStage()
        self.ex_mem = PipelineStage()
        self.mem_wb = PipelineStage()
        self.cycle = 0
        
    def pipeline_step(self, l1i, l1d):
        """Executa um ciclo completo do pipeline"""
        self.cycle += 1
        
        # Processar estágios em ordem reversa (WB -> IF)
        self.writeback()
        self.memory(l1d)
        self.execute()
        self.decode()
        self.fetch(l1i)
        
        # Avançar registradores entre estágios
        self.advance_pipeline()
    
    def fetch(self, l1i):
        """IF: Busca instrução da cache L1I"""
        instruction, cycles = l1i.access(self.pc)
        self.if_id.instruction = instruction
        self.if_id.pc = self.pc
        self.if_id.is_bubble = False
        self.pc += 4  # Próxima instrução
        return cycles
    
    def decode(self):
        """ID: Decodifica instrução e lê registradores"""
        if self.if_id.is_bubble:
            return
        
        inst = self.if_id.instruction
        # Extrai campos (simplificado)
        opcode = inst & 0x7F
        rd = (inst >> 7) & 0x1F
        rs1 = (inst >> 15) & 0x1F
        rs2 = (inst >> 20) & 0x1F
        
        # Lê registradores
        self.id_ex.rs1_val = self.registers[rs1]
        self.id_ex.rs2_val = self.registers[rs2]
        self.id_ex.rd = rd
        self.id_ex.instruction = inst
        self.id_ex.is_bubble = False
    
    def execute(self):
        """EX: Executa operação na ALU"""
        if self.id_ex.is_bubble:
            return
        
        # Operação simplificada (ADD exemplo)
        result = self.id_ex.rs1_val + self.id_ex.rs2_val
        self.ex_mem.alu_result = result
        self.ex_mem.rd = self.id_ex.rd
        self.ex_mem.is_bubble = False
    
    def memory(self, l1d):
        """MEM: Acessa cache L1D para loads/stores"""
        if self.ex_mem.is_bubble:
            return
        
        # Exemplo: load
        data, cycles = l1d.access(self.ex_mem.alu_result)
        self.mem_wb.mem_data = data
        self.mem_wb.alu_result = self.ex_mem.alu_result
        self.mem_wb.rd = self.ex_mem.rd
        self.mem_wb.is_bubble = False
        return cycles
    
    def writeback(self):
        """WB: Escreve resultado no banco de registradores"""
        if self.mem_wb.is_bubble or self.mem_wb.rd == 0:
            return
        
        self.registers[self.mem_wb.rd] = self.mem_wb.alu_result
    
    def advance_pipeline(self):
        """Avança instruções pelos estágios"""
        # Em implementação real, copiar registradores
        pass


# ============================================================================
# 2. DETECÇÃO E RESOLUÇÃO DE HAZARDS
# ============================================================================

class HazardUnit:
    @staticmethod
    def detect_data_hazard(id_ex, ex_mem, mem_wb):
        """Detecta dependências RAW"""
        hazards = []
        
        # Verifica se ID/EX precisa de dado que está em EX/MEM
        if ex_mem.rd != 0 and (ex_mem.rd == id_ex.rs1_val or ex_mem.rd == id_ex.rs2_val):
            hazards.append(('EX', ex_mem.rd))
        
        # Verifica se ID/EX precisa de dado que está em MEM/WB
        if mem_wb.rd != 0 and (mem_wb.rd == id_ex.rs1_val or mem_wb.rd == id_ex.rs2_val):
            hazards.append(('MEM', mem_wb.rd))
        
        return hazards
    
    @staticmethod
    def check_forwarding(hazards):
        """Verifica se forwarding resolve o hazard"""
        for source, reg in hazards:
            if source in ['EX', 'MEM']:
                return True  # Pode fazer bypass
        return False
    
    @staticmethod
    def apply_forwarding(id_ex, ex_mem, mem_wb):
        """Implementa bypass de dados"""
        # Forward de EX/MEM para EX
        if ex_mem.rd != 0 and ex_mem.rd == id_ex.rs1_val:
            id_ex.rs1_val = ex_mem.alu_result
        
        # Forward de MEM/WB para EX
        if mem_wb.rd != 0 and mem_wb.rd == id_ex.rs1_val:
            id_ex.rs1_val = mem_wb.alu_result
    
    @staticmethod
    def insert_stall(pipeline):
        """Insere bolha no pipeline"""
        # Transforma ID/EX em bolha
        pipeline.id_ex.is_bubble = True
        # Mantém IF/ID (não avança PC)
        pipeline.pc -= 4


# ============================================================================
# 3. PREDIÇÃO DE DESVIOS
# ============================================================================

class BranchPredictor:
    def __init__(self, mode='not-taken'):
        self.mode = mode
        self.predictor_table = {}  # Tabela de predição (2-bits)
    
    def predict_branch(self, pc):
        """Faz predição de desvio"""
        if self.mode == 'not-taken':
            return False  # Sempre não desvia
        
        elif self.mode == '2-bit':
            state = self.predictor_table.get(pc, 0)  # 0=strongly NT, 3=strongly T
            return state >= 2
        
        return False
    
    def update_predictor(self, pc, taken):
        """Atualiza preditor após resolução"""
        if self.mode == '2-bit':
            state = self.predictor_table.get(pc, 1)  # Inicia em weakly NT
            
            if taken:
                state = min(state + 1, 3)  # Incrementa (max 3)
            else:
                state = max(state - 1, 0)  # Decrementa (min 0)
            
            self.predictor_table[pc] = state
    
    def resolve_branch(self, condition, target_pc, next_pc):
        """Determina resultado real do branch"""
        taken = condition  # Simplificado
        correct_pc = target_pc if taken else next_pc
        return taken, correct_pc


# ============================================================================
# 4. INTERFACE COM CACHE
# ============================================================================

class CacheInterface:
    def __init__(self, cache):
        self.cache = cache
        self.pending_cycles = 0
    
    def fetch_instruction(self, address):
        """Solicita instrução à L1I"""
        return self.cache.access(address, is_read=True)
    
    def load_data(self, address):
        """Lê dado da L1D"""
        data, cycles = self.cache.access(address, is_read=True)
        return data, cycles
    
    def store_data(self, address, data):
        """Escreve dado na L1D"""
        cycles = self.cache.access(address, is_read=False, data=data)
        return cycles


# ============================================================================
# 5. SISTEMA DE CACHE
# ============================================================================

class CacheLine:
    def __init__(self):
        self.valid = False
        self.tag = 0
        self.data = 0
        self.dirty = False
        self.lru_counter = 0

class Cache:
    def __init__(self, size=1024, line_size=64, associativity=2, 
                 hit_latency=1, miss_penalty=10):
        self.size = size
        self.line_size = line_size
        self.associativity = associativity
        self.num_sets = size // (line_size * associativity)
        self.hit_latency = hit_latency
        self.miss_penalty = miss_penalty
        
        # Inicializa cache
        self.sets = [[CacheLine() for _ in range(associativity)] 
                     for _ in range(self.num_sets)]
    
    def cache_lookup(self, address):
        """Busca endereço na cache"""
        set_idx = (address // self.line_size) % self.num_sets
        tag = address // (self.line_size * self.num_sets)
        
        for way in self.sets[set_idx]:
            if way.valid and way.tag == tag:
                return True, way.data  # HIT
        
        return False, None  # MISS
    
    def cache_allocate(self, address, data):
        """Aloca linha na cache"""
        set_idx = (address // self.line_size) % self.num_sets
        tag = address // (self.line_size * self.num_sets)
        
        # Procura entrada inválida ou usa eviction
        victim_way = self.cache_evict(set_idx)
        
        self.sets[set_idx][victim_way].valid = True
        self.sets[set_idx][victim_way].tag = tag
        self.sets[set_idx][victim_way].data = data
        self.sets[set_idx][victim_way].dirty = False
    
    def cache_evict(self, set_idx):
        """Remove linha usando LRU"""
        # Encontra way com menor contador LRU
        min_lru = float('inf')
        victim = 0
        
        for i, way in enumerate(self.sets[set_idx]):
            if not way.valid:
                return i  # Usa entrada inválida
            if way.lru_counter < min_lru:
                min_lru = way.lru_counter
                victim = i
        
        return victim
    
    def access(self, address, is_read=True, data=None):
        """Processa acesso à cache"""
        hit, cached_data = self.cache_lookup(address)
        
        if hit:
            return cached_data, self.hit_latency
        else:
            # Miss: busca da memória
            mem_data = data if data else self.memory_access(address)
            self.cache_allocate(address, mem_data)
            return mem_data, self.hit_latency + self.miss_penalty
    
    def memory_access(self, address):
        """Simula acesso à memória principal"""
        # Simula leitura da DRAM
        return 0xDEADBEEF  # Dado fictício


# ============================================================================
# 6. BANCO DE REGISTRADORES
# ============================================================================

class RegisterFile:
    def __init__(self):
        self.registers = [0] * 32
        self.registers[0] = 0  # x0 sempre 0
    
    def read_register(self, reg_num):
        """Lê valor de registrador"""
        if 0 <= reg_num < 32:
            return self.registers[reg_num]
        return 0
    
    def write_register(self, reg_num, value):
        """Escreve valor em registrador"""
        if 0 < reg_num < 32:  # x0 não pode ser escrito
            self.registers[reg_num] = value & 0xFFFFFFFF  # 32 bits


# ============================================================================
# 7. DECODIFICAÇÃO DE INSTRUÇÕES
# ============================================================================

class InstructionType(Enum):
    R_TYPE = 1
    I_TYPE = 2
    S_TYPE = 3
    B_TYPE = 4
    J_TYPE = 5

class Decoder:
    @staticmethod
    def decode_instruction(instruction):
        """Decodifica instrução RISC-V"""
        opcode = instruction & 0x7F
        rd = (instruction >> 7) & 0x1F
        funct3 = (instruction >> 12) & 0x7
        rs1 = (instruction >> 15) & 0x1F
        rs2 = (instruction >> 20) & 0x1F
        funct7 = (instruction >> 25) & 0x7F
        
        return {
            'opcode': opcode,
            'rd': rd,
            'funct3': funct3,
            'rs1': rs1,
            'rs2': rs2,
            'funct7': funct7
        }
    
    @staticmethod
    def identify_instruction_type(opcode):
        """Classifica tipo de instrução"""
        if opcode == 0x33:  # ADD, SUB, AND, OR, XOR, SLT
            return InstructionType.R_TYPE
        elif opcode in [0x13, 0x03]:  # ADDI, LW
            return InstructionType.I_TYPE
        elif opcode == 0x23:  # SW
            return InstructionType.S_TYPE
        elif opcode == 0x63:  # BEQ, BNE
            return InstructionType.B_TYPE
        elif opcode in [0x6F, 0x67]:  # JAL, JALR
            return InstructionType.J_TYPE
        return None
    
    @staticmethod
    def extract_immediate(instruction, inst_type):
        """Extrai imediato baseado no tipo"""
        if inst_type == InstructionType.I_TYPE:
            imm = (instruction >> 20) & 0xFFF
            # Sign extend
            if imm & 0x800:
                imm |= 0xFFFFF000
            return imm
        
        elif inst_type == InstructionType.S_TYPE:
            imm = ((instruction >> 7) & 0x1F) | ((instruction >> 20) & 0xFE0)
            if imm & 0x800:
                imm |= 0xFFFFF000
            return imm
        
        # Outros tipos...
        return 0


# ============================================================================
# 8. ESTRUTURAS DE CONTROLE
# ============================================================================

class PipelineController:
    def __init__(self):
        self.pipeline = Pipeline()
        self.l1i = Cache(size=8192, hit_latency=1)  # 8KB L1I
        self.l1d = Cache(size=8192, hit_latency=2)  # 8KB L1D
        self.hazard_unit = HazardUnit()
        self.branch_predictor = BranchPredictor(mode='2-bit')
        self.total_cycles = 0
    
    def clock_cycle(self):
        """Gerencia ciclo de relógio global"""
        self.total_cycles += 1
        self.pipeline.pipeline_step(self.l1i, self.l1d)
    
    def is_pipeline_empty(self):
        """Verifica se pipeline terminou execução"""
        return (self.pipeline.if_id.is_bubble and
                self.pipeline.id_ex.is_bubble and
                self.pipeline.ex_mem.is_bubble and
                self.pipeline.mem_wb.is_bubble)
    
    def flush_pipeline(self):
        """Limpa pipeline após branch misprediction"""
        self.pipeline.if_id.is_bubble = True
        self.pipeline.id_ex.is_bubble = True
        self.pipeline.ex_mem.is_bubble = True
        # MEM/WB não é flushed (já passou do branch)


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    controller = PipelineController()
    
    print("Simulador de Pipeline 5 estágios inicializado")
    print(f"L1I: {controller.l1i.size} bytes, {controller.l1i.associativity}-way")
    print(f"L1D: {controller.l1d.size} bytes, {controller.l1d.associativity}-way")
    
    # Simula alguns ciclos
    for i in range(10):
        controller.clock_cycle()
        print(f"Ciclo {controller.total_cycles}: Pipeline executando...")
    
    print(f"\nTotal de ciclos simulados: {controller.total_cycles}")
    print(f"Pipeline vazio: {controller.is_pipeline_empty()}")
