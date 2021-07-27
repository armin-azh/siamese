package main

import (
	"./controller"
	 "./controller/identifier/os/windows"
	"fmt"
	_ "github.com/shirou/gopsutil/cpu"
	"github.com/yumaojun03/dmidecode"
	_ "github.com/yumaojun03/dmidecode"
	"os"
)

func main(){

	//var filename = "armin.zip"
	//var target = "data/armin.zip"
	//var outputPath = "data/temp"
    //filenames,err:=compressor.Unzip(target,outputPath)
	//controller.GetCurrentPath()
    fmt.Println(controller.GetCurrentPath())
    //fmt.Println(filenames)
    fmt.Println(os.Getpagesize())

    dmi,_:= dmidecode.New()

	infos,_:=dmi.Processor()
	fmt.Println(infos)

	a, _ :=windows.CpuInfo()
	b,_:=windows.DmiInfo()

	fmt.Println(a)
	fmt.Println(b)

}