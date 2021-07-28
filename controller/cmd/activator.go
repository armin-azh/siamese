package main

import (
	".."
	"../identifier"
	"os"
	"path/filepath"
	"time"
)



func main(){
	device,err:=identifier.CreateNewDevice()
	if err!=nil{
		panic(err)
	}
	args:=os.Args
	var outputPath string
	yamlFilename:="activation.yaml"
	mp:=make(map[string]string)
	mp["activation_code"] = device.ProtectMachineID()
	mp["created"] = time.Now().String()
	if len(args)<3 || (len(args)>=3 && args[1]!="-o"){
		outputPath = yamlFilename
		err=controller.WriteYamlFile(outputPath,mp)
		if err!=nil{
			panic(err)
		}
	}else{
		outputPath = args[2]
		_,err=os.Stat(outputPath)
		if os.IsNotExist(err){
			err=os.MkdirAll(outputPath,0777)
			if err!=nil{
				panic(err)
			}
		}else{
			fullFilePath:=filepath.Join(outputPath,yamlFilename)
			err=controller.WriteYamlFile(fullFilePath,mp)
			if err!=nil{
				panic(err)
			}
		}

	}
}
