#include <iostream>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include "hlo.pb.h"

int 
add(int i, int j)
{
    return i + j;
}

static llvm::orc::ThreadSafeModule 
compile_comp(const xla::HloComputationProto& comp)
{
    auto Context = std::make_unique<llvm::LLVMContext>();
    auto M = std::make_unique<llvm::Module>(comp.name(), *Context);

    /*
    llvm::Function *F = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getFloatTy(*Context),
            {llvm::Type::getFloatTy(*Context)},
            false), llvm::Function::ExternalLinkage, comp.name(), M.get());
*/
    llvm::Function *F = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getInt32Ty(*Context),
            {llvm::Type::getInt32Ty(*Context)},
            false), llvm::Function::ExternalLinkage, comp.name(), M.get());

    llvm::BasicBlock *BB = llvm::BasicBlock::Create(*Context, "EntryBlock", F);

    llvm::IRBuilder<> builder(BB);

    std::unordered_map<int64_t,llvm::Value *> locals;

    for (auto& ins : comp.instructions()) {
        if (ins.opcode() == "parameter") {
            std::cout << "adding param " << ins.id() << std::endl;
            // llvm::Argument *a = &*(F->arg_begin() + ins.parameter_number());
            // locals[ins.id()] = a;
        } else if (ins.opcode() == "constant") {
            std::cout << "adding constant" << std::endl;
        } else if (ins.opcode() == "add") {
            std::cout << "ignoring add" << std::endl;
        } else if (ins.opcode() == "sub") {
            std::cout << "ignoring sub" << std::endl;
        } else if (ins.opcode() == "multiply") {
            std::cout << "ignoring mul" << std::endl;
        } else if (ins.opcode() == "divide") {
            std::cout << "ignoring div" << std::endl;
        } else if (ins.opcode() == "exponential") {
            std::cout << "ignoring mul" << std::endl;
        } else if (ins.opcode() == "tuple") {
            std::cout << "ignoring tuple" << std::endl;
        }
    }

    // llvm::ConstantFP *Foo = llvm::ConstantFP::get(*Context, llvm::APFloat(1.234f));
    // llvm::Value *LoadFoo = builder.CreateLoad(llvm::Type::getFloatTy(*Context), Foo);

    llvm::Value *One = builder.getInt32(1);

    llvm::Argument *ArgX = &*F->arg_begin();
    ArgX->setName("AnArg");
    llvm::Value *Add = builder.CreateAdd(One, ArgX);

    builder.CreateRet(Add);

    return llvm::orc::ThreadSafeModule(std::move(M), std::move(Context));
}

void
print_shape(const xla::ShapeProto& sh)
{
    std::cout << xla::PrimitiveType_Name(sh.element_type());
    if (sh.element_type() == xla::PrimitiveType::TUPLE) {
        std::cout << "(";
        for (int i = 0; i < sh.tuple_shapes().size(); i++) {
            print_shape(sh.tuple_shapes()[i]);
            if (i < sh.tuple_shapes().size() - 1) std::cout << ",";
        }
        std::cout << ")";
    }
    auto& dd = sh.is_dynamic_dimension();
    for (int i = 0; i < sh.dimensions().size(); i++) {
        std::cout << "[";
        if (dd.size() > 0 && dd[i]) std::cout << "^";
        std::cout << sh.dimensions()[i];
        std::cout << "]";
    }
}

void
take(std::string s)
{
    xla::HloModuleProto m;
    m.ParseFromString(s);
    std::cout << "parsed:" << std::endl;
    std::cout << "name: " << m.name() << std::endl;
    std::cout << "entry_name: " << m.entry_computation_name() << std::endl;
    std::cout << "entry_id: " << m.entry_computation_id() << std::endl;
    std::cout << "computations: " << m.computations().size() << std::endl;
    for (auto& comp : m.computations()) {
        std::cout << "name:" << comp.name() << std::endl;

        auto& psh = comp.program_shape();
        std::cout << "\tshape:" << std::endl;
        std::cout << "\t\tparam names: ";
        for (int i = 0; i < psh.parameter_names().size(); i++) {
            std::cout << psh.parameter_names()[i];
            if (i < psh.parameter_names().size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        std::cout << "\t\ttakes: (";
        for (int i = 0; i < psh.parameters().size(); i++) {
            print_shape(psh.parameters()[i]);
            if (i < psh.parameters().size() - 1) std::cout << ",";
        }
        std::cout << ")" << std::endl;
        std::cout << "\t\treturns: ";
        print_shape(psh.result());
        std::cout << std::endl;

        std::cout << "\tinstrs: " << comp.instructions().size() << std::endl;
        for (auto& ins : comp.instructions()) {
            std::cout << "\tname:" << ins.name() << std::endl;
            std::cout << "\t\topcode:" << ins.opcode();
            if (ins.opcode() == "parameter") {
                std::cout << " param(" << ins.parameter_number() << ")";
            } else if (ins.opcode() == "constant") {
                auto& lit = ins.literal();
                std::cout << " ";
                print_shape(lit.shape());
            } else {
                std::cout << " operands:";
                for (auto& o : ins.operand_ids()) {
                    std::cout << o << " ";
                }
            }
            std::cout << std::endl;
        }
    }

    int argc = 1;
    const char *progname = "jaxopt_native";
    const char **argv = &progname;
    llvm::InitLLVM X(argc, argv);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    llvm::ExitOnError ExitOnErr;

    std::cout << "jitting" << std::endl;
    auto J = ExitOnErr(llvm::orc::LLJITBuilder().create());
    auto M = compile_comp(m.computations()[0]);
    ExitOnErr(J->addIRModule(std::move(M)));

    auto CompSym = ExitOnErr(J->lookup("xla_computation_tanh.12"));
    // float (*CompFn)(float) = (float (*)(float)) CompSym.getAddress();
    int32_t (*CompFn)(int32_t) = (int32_t (*)(int32_t)) CompSym.getAddress();

    std::cout << "running" << std::endl;
    // float result = CompFn(1.2);
    // std::cout << "tanh(1.2) == " << result << std::endl;
    int32_t result = CompFn(2);
    std::cout << "tanh(1) == " << result << std::endl;

    std::cout << "OK" << std::endl;
}

PYBIND11_MODULE(jaxopt_native, m) {
    m.def("add", &add);
    m.def("take", &take);
}
