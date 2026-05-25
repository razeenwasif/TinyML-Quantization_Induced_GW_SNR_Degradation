### Literature Review: TensorFlow Lite Micro for TinyML Systems

The 2021 paper by David et al. introduces TensorFlow Lite Micro (TF Micro), an open-source machine learning inference framework explicitly designed to bridge the gap between deep learning and severely resource-constrained embedded systems. As the TinyML field expands, deploying models to microcontrollers (MCUs) and Digital Signal Processors (DSPs) faces extreme hardware limitations: power budgets in the milliwatts and working memory restricted to a few hundred kilobytes.

**Key Design Principles and Innovations:**

- **Interpreter-Based Architecture:** Instead of using ahead-of-time code generation (which is rigid and hard to update), TF Micro uses an interpreter-based approach. The authors demonstrate that because the latency is overwhelmingly dominated by the underlying linear-algebra computations, the overhead of the interpreter itself is negligible (less than 1% for most models).
    
- **No Dynamic Memory Allocation:** Embedded systems often lack virtual memory and dynamic memory managers. TF Micro circumvents this by requiring the application to provide a single, pre-allocated, fixed-size memory "arena". A "Memory Planner" then uses a two-stack strategy and bin-packing algorithms at initialization to calculate the optimal layout for intermediate tensors, maximizing reuse and ensuring the model fits within the strict SRAM limits.
    
- **The "Bag of Files" Principle:** To combat the heavy fragmentation of the embedded market—where vendors use highly specialized, proprietary toolchains—TF Micro avoids complex build systems. It is designed so developers can simply drop the core C++ source files into any IDE and compile them without dependencies on external libraries or operating systems.
    
- **Vendor-Specific Kernel Optimizations:** While the core framework is hardware-agnostic, it allows hardware vendors (like ARM or Cadence) to swap out the default reference kernels for highly optimized, platform-specific implementations (e.g., ARM's CMSIS-NN). The evaluation shows these optimizations yield massive speedups, making complex models viable on bare-metal hardware.