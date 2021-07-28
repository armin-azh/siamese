package identifier

import "github.com/denisbrodbeck/machineid"

type Device struct {
	machineID string
	machineHashID string
}

func CreateNewDevice()(*Device,error){
	nID,err:=machineid.ID()
	if err!=nil{
		return nil,err
	}
	pID,err:=machineid.ProtectedID("face_recognition")
	if err!=nil{
		return nil,err
	}
	device:=Device{machineID: nID,machineHashID: pID}

	return &device,nil
}


func (dv *Device)MachineID()string{
	return dv.machineID
}

func (dv *Device)ProtectMachineID()string{
	return dv.machineHashID
}