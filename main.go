package main

import (
	"./controller"
	"fmt"
	"github.com/shirou/gopsutil/cpu"
	_ "github.com/shirou/gopsutil/cpu"
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

    cpuList,err:=cpu.Info()
    if err!=nil{
    	panic(err)
	}

	fmt.Println(cpuList)

}