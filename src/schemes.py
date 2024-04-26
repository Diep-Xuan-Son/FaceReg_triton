from typing import List, Union
from pydantic import BaseModel
from fastapi import Query, File, UploadFile

class Person(BaseModel):
	code: str = Query("001099008839", description="ID of person, ID is unique")
	name: str = Query("Son", description="Name of person")
	birthday: str = Query("29/04/1999", description="Birthday of person")
	images:	List[UploadFile] = File(...)
