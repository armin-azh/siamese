package os

import "strconv"

type DMI struct {
	sysVendor string
	biosVendor string
	biosDescription string
	cpuManufacturer string
	cpuCore uint
}

func CreateNewDmi(sysV string,biosV string,biosD string,cpuM string,cpuC uint)*DMI{
	dm:=DMI{sysVendor: sysV,biosVendor: biosV,biosDescription: biosD,cpuManufacturer: cpuM,cpuCore: cpuC}
	return &dm
}

func (dm *DMI) Info()string{
	return dm.sysVendor+"_"+dm.biosDescription+"_"+dm.biosVendor+"_"+dm.cpuManufacturer+"_"+strconv.Itoa(int(dm.cpuCore))
}

func (dm *DMI) SysVendor () string{
	return dm.sysVendor
}

func (dm *DMI) BioVendor()string{
	return dm.biosVendor
}

func (dm *DMI) BioDescription()string{
	return dm.biosDescription
}

func (dm *DMI) CpuManufacturer()string{
	return dm.cpuManufacturer
}

func (dm *DMI) CpuCore()uint{
	return dm.cpuCore
}
