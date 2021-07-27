package os

const (
	Docker = "Docker" // docker virtualization
	Lxc = "LXC" //operating system level virtualization
	Nada = "None"
)


type ExeEnv struct {
	cCpuInfo CPU
	cDmiInfo DMI
	cContainerType string
}


func CreateExeEnv(cpInfo CPU,dmInfo DMI,container string)*ExeEnv{
	return &ExeEnv{cCpuInfo: cpInfo,cDmiInfo: dmInfo,cContainerType: container}
}

func (ex *ExeEnv)isDocker()bool{
	// check if the execution env is docker container
	return ex.cContainerType == Docker
}

func (ex *ExeEnv)isContainer()bool{
	// check if the execution env is a container or not
	return ex.cContainerType!=Nada
}

func (ex *ExeEnv)isLxc()bool{
	// check if the execution env is lxc container
	return ex.cContainerType == Lxc
}