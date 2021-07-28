package controller

import (
	"gopkg.in/yaml.v3"
	"io/ioutil"
	"os"
)

func GetCurrentPath()string{
	s,err:=os.Getwd()
	if err!=nil{
		panic(err)
	}
	return s

}


func WriteYamlFile(output string,data map[string]string)error{
	// write map datatype to YAML file in specific output
	ans,err:=yaml.Marshal(data)
	if err!=nil{
		return err
	}
	err = ioutil.WriteFile(output,ans,0777)
	if err!=nil{
		return err
	}
	return nil
}
