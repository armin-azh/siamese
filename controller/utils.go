package controller

import "os"

func GetCurrentPath()string{
	s,err:=os.Getwd()
	if err!=nil{
		panic(err)
	}

	return s

}
