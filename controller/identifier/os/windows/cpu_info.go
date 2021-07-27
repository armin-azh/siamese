package windows

import (
	osp ".."
	"github.com/shirou/gopsutil/cpu"
)

func CpuInfo() ([]osp.CPU,error){
	var cpuList []osp.CPU
	cpuInfoList,err:=cpu.Info()
	if err!=nil{
		return cpuList,err
	}

	// initiate cpu information
	for _,cp := range cpuInfoList{
		newCPU:=osp.CreateNewCpu(cp.VendorID,cp.ModelName)
		cpuList = append(cpuList,*newCPU)
	}

	return cpuList,nil
}