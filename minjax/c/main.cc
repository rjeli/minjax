#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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

static llvm::orc::ThreadSafeModule 
compile_comp(const xla::HloComputationProto& comp, std::string name)
{
    auto Context = std::make_unique<llvm::LLVMContext>();
    auto M = std::make_unique<llvm::Module>("test", *Context);

    llvm::Function *F = llvm::Function::Create(
        llvm::FunctionType::get(
            llvm::Type::getFloatTy(*Context),
            {llvm::Type::getFloatTy(*Context)},
            false), 
        llvm::Function::ExternalLinkage, name, M.get());

    llvm::BasicBlock *BB = llvm::BasicBlock::Create(*Context, "EntryBlock", F);

    llvm::IRBuilder<> builder(BB);

    std::unordered_map<int64_t,llvm::Value *> locals;

    for (auto& ins : comp.instructions()) {
        if (ins.opcode() == "parameter") {
            std::cout << "adding param " << ins.id() << " " << ins.parameter_number() << std::endl;
            int nargs = F->arg_end() - F->arg_begin();
            assert(ins.parameter_number() >= 0);
            assert(ins.parameter_number() < nargs);
            llvm::Argument *a = &*(F->arg_begin() + ins.parameter_number());
            locals[ins.id()] = a;
        } else if (ins.opcode() == "constant") {
            std::cout << "adding constant" << std::endl;
            auto& lit = ins.literal();
            auto& litsh = lit.shape();
            if (litsh.dimensions().size() == 0 && litsh.element_type() == xla::PrimitiveType::F32) {
                float v = lit.f32s()[0];
                std::cout << "  " << v << std::endl;
                llvm::Value *vv = llvm::ConstantFP::get(llvm::Type::getFloatTy(*Context), v);
                locals[ins.id()] = vv;
            } else {
                std::cout << "  unsupported shape: ";
                print_shape(litsh);
                std::cout << std::endl;
            }
        } else if (ins.opcode() == "add") {
            std::cout << "adding add" << std::endl;
            llvm::Value *o1 = locals[ins.operand_ids()[0]];
            llvm::Value *o2 = locals[ins.operand_ids()[1]];
            llvm::Value *res = builder.CreateFAdd(o1, o2);
            locals[ins.id()] = res;
        } else if (ins.opcode() == "subtract") {
            std::cout << "adding sub" << std::endl;
            llvm::Value *o1 = locals[ins.operand_ids()[0]];
            llvm::Value *o2 = locals[ins.operand_ids()[1]];
            llvm::Value *res = builder.CreateFSub(o1, o2);
            locals[ins.id()] = res;
        } else if (ins.opcode() == "multiply") {
            std::cout << "adding mul" << std::endl;
            llvm::Value *o1 = locals[ins.operand_ids()[0]];
            llvm::Value *o2 = locals[ins.operand_ids()[1]];
            llvm::Value *res = builder.CreateFMul(o1, o2);
            locals[ins.id()] = res;
        } else if (ins.opcode() == "divide") {
            std::cout << "adding div" << std::endl;
            llvm::Value *o1 = locals[ins.operand_ids()[0]];
            llvm::Value *o2 = locals[ins.operand_ids()[1]];
            llvm::Value *res = builder.CreateFDiv(o1, o2);
            locals[ins.id()] = res;
        } else if (ins.opcode() == "exponential") {
            std::cout << "ignoring exp" << std::endl;
        } else if (ins.opcode() == "tuple") {
            std::cout << "adding tuple" << std::endl;
            llvm::Value *o1 = locals[ins.operand_ids()[0]];
            std::cout << "  o1: " << o1 << std::endl;
            builder.CreateRet(o1);
        } else {
            std::cout << "unrecognized opcode: " << ins.opcode() << std::endl;
        }
    }

    return llvm::orc::ThreadSafeModule(std::move(M), std::move(Context));
}

void
dump(std::string s)
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
}

class Problem {
public:
    void solve(std::string fn, std::string grad, py::array_t<float> arg) {
        std::cout << "solving" << std::endl;

        std::cout << "initializing llvm" << std::endl;
        int argc = 1;
        const char *progname = "jaxopt_native";
        const char **argv = &progname;
        llvm::InitLLVM X(argc, argv);
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::ExitOnError ExitOnErr;
        ExitOnErr.setBanner("EXITONERR: ");

        std::cout << "parsing modules" << std::endl;
        xla::HloModuleProto fn_mod;
        fn_mod.ParseFromString(fn);
        assert(fn_mod.computations().size() == 1);
        xla::HloModuleProto grad_mod;
        grad_mod.ParseFromString(grad);
        assert(grad_mod.computations().size() == 1);

        std::cout << "jitting" << std::endl;
        auto J = ExitOnErr(llvm::orc::LLJITBuilder().create());
        auto Fn = compile_comp(fn_mod.computations()[0], "fn");
        ExitOnErr(J->addIRModule(std::move(Fn)));
        auto Grad = compile_comp(grad_mod.computations()[0], "grad");
        ExitOnErr(J->addIRModule(std::move(Grad)));

        std::cout << "getting syms" << std::endl;
        auto FnSym = ExitOnErr(J->lookup("fn"));
        float (*FnPtr)(float) = (float (*)(float)) FnSym.getAddress();
        auto GradSym = ExitOnErr(J->lookup("grad"));
        float (*GradPtr)(float) = (float (*)(float)) GradSym.getAddress();

        std::cout << "running" << std::endl;
        float res = FnPtr(1.5);
        float gres = GradPtr(1.5);
        std::cout << "x=1.5 fn(x)=" << res << " grad(fn(x))=" << gres << std::endl;

        // simple gradient descent
        float *x = arg.mutable_data();
        float err, gerr;
        for (int i = 0; i < 100; i++) {
            std::cout << "iter " << i << std::endl;
            std::cout << "  x: " << *x << std::endl;
            err = FnPtr(*x);
            std::cout << "  err: " << err << std::endl;
            if (err < 1e-3) break;
            gerr = GradPtr(*x);
            std::cout << "  grad: " << gerr << std::endl;
            *x += -1e-1 * gerr;
        }

        std::cout << "OK " << *x << std::endl;
    }
};

PYBIND11_MODULE(minjax_c, m) {
    m.def("add", &add);
    m.def("dump", &dump);
    py::class_<Problem>(m, "Problem")
        .def(py::init<>())
        .def("solve", &Problem::solve);
}
