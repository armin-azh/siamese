package windows

import (
	osp ".."
	"github.com/yumaojun03/dmidecode"
)

func DmiInfo()(osp.DMI,error){

	dmi,err:= dmidecode.New()
	if err!=nil{
		tmp:= osp.CreateNewDmi("","","","",0)
		return *tmp,err
	}

	bios,err:=dmi.BIOS()
	if err!=nil{
		tmp:= osp.CreateNewDmi("","","","",0)
		return *tmp,err
	}

	biosV:= bios[0].Vendor
	biosD:= bios[0].BIOSVersion

	sys,err:=dmi.BaseBoard()
	if err!=nil{
		tmp:= osp.CreateNewDmi("","","","",0)
		return *tmp,err
	}

	sysV:= sys[0].Manufacturer+"("+sys[0].SerialNumber+")"

	cpu,err:=dmi.Processor()
	if err!=nil{
		tmp:= osp.CreateNewDmi("","","","",0)
		return *tmp,err
	}

	cpuM:=cpu[0].Manufacturer
	cpuC:=cpu[0].CoreCount

	dmiAns:= osp.CreateNewDmi(sysV,biosV,biosD,cpuM,uint(cpuC))
	return *dmiAns,nil

}
