echo "================================================================="
echo "               AI ENGINEER SYSTEM DIAGNOSTIC REPORT              "
echo "================================================================="

# 1. OS & KERNEL INFORMATION
echo -e "\n[1] OS & Environment Framework:"
if command -v hostnamectl &> /dev/null; then
    hostnamectl | grep -E "Operating System|Kernel|Architecture" | sed 's/^/  /'
else
    echo "  OS: Linux"
    echo "  Kernel: $(uname -r)"
fi
echo "  Shell: $SHELL ($BASH_VERSION)"

# 2. CPU COMPUTE CAPABILITIES (AI Extensions focus)
echo -e "\n[2] CPU Compute Configuration:"
if command -v lscpu &> /dev/null; then
    echo "  Model: $(lscpu | grep 'Model name:' | sed 's/Model name:[[:space:]]*//')"
    echo "  Cores/Threads: $(lscpu | grep -E '^CPU\(s\):' | awk '{print $2}') Cores total"
    # Scrape for architectural acceleration extensions critical for deep learning frameworks
    isa_flags=$(lscpu | grep 'Flags:' | tr ' ' '\n' | grep -E -i 'avx|amx|fma|f16c' | sort -u | tr '\n' ' ')
    echo "  AI Instruction Extensions: ${isa_flags:-None detected}"
fi

# 3.  iGPU & COMPUTE (ROCm / Sysfs)
VRAM_TOTAL_FILE=$(ls /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | head -n 1)
VRAM_USED_FILE=$(ls /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -n 1)

if [ -n "$VRAM_TOTAL_FILE" ] && [ -n "$VRAM_USED_FILE" ]; then
    raw_total=$(cat "$VRAM_TOTAL_FILE")
    raw_used=$(cat "$VRAM_USED_FILE")
    vram_total_gib=$(awk "BEGIN {printf \"%.2f\", $raw_total/1024/1024/1024}")
    vram_used_gib=$(awk "BEGIN {printf \"%.2f\", $raw_used/1024/1024/1024}")
    vram_pct=$(awk "BEGIN {printf \"%.1f\", ($raw_used/$raw_total)*100}")
    
    # Query exact hardware chip info from lspci
    gpu_name=$(lspci -nnk | grep -i -E "vga|3d" | grep -i "AMD" | head -n 1 | sed -E 's/.*\[AMD\/ATI\] //')
    echo "  GPU Model: ${gpu_name}"
    echo "  Allocated VRAM Pool: $vram_total_gib GiB"
    echo "  VRAM Active Footprint: $vram_used_gib GiB ($vram_pct%)"
else
    echo "  GPU: AMD Telemetry unreadable (Check amdgpu drivers)"
fi

# Check for active ROCm environments
if command -v rocm-smi &> /dev/null; then
    echo "  ROCm Driver Stack: Detected"
    echo "  ROCm Agent Info: $(rocm-smi --showid | grep -i 'device' | head -n 1 | awk '{print $4,$5}')"
elif [ -d "/opt/rocm" ]; then
    echo "  ROCm Toolkit Path: Found (/opt/rocm directory exists, binaries not in PATH)"
else
    echo "  ROCm Status: Not globally initialized (Required for local PyTorch/HIP acceleration)"
fi

# 4. SYSTEM RAM DATA (For heavy context windows / quantized sharding)
echo -e "\n[4] System Unified Memory Capacity:"
free -h | awk '
    /Mem:/ {print "  Total Physical RAM: " $2 "\n  Allocated/Used:     " $3 "\n  Free/Cached:        " $7}
    /Swap:/ {print "  Swap Space:         " $2 " (" $3 " used)"}
'

# 5. STORAGE SPECIFICATIONS (Crucial for large tensor datasets / weights cache)
echo -e "\n[5] Storage & Datasets File System Space:"
df -h / | awk 'NR==2 {print "  Root Partition (/): " $3 " Used / " $2 " Total Space Available (" $5 " Utilised) - Type: " $6}'
if [ -d "$HOME/.cache/huggingface" ]; then
    hf_size=$(du -sh "$HOME/.cache/huggingface" 2>/dev/null | awk '{print $1}')
    echo "  HuggingFace Weight Cache: $hf_size"
fi

echo -e "\n================================================================="