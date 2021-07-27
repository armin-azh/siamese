package os


type CPU struct {
	cpuVendor string
	cpuBrand string
}

func CreateNewCpu(vendor string,brand string)*CPU{
	cp := CPU{cpuBrand: brand,cpuVendor: vendor}
	return &cp
}

func (cp *CPU)Info() string {
	return cp.cpuBrand+"_"+cp.cpuVendor
}

func (cp *CPU)Vendor()string{
	return cp.cpuVendor
}

func (cp *CPU)Brand()string{
	return cp.cpuBrand
}


