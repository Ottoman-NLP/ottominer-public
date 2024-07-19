import module1 as md
import sys 


print(type(md))
import types 
print(isinstance(md, types.ModuleType))
print(hex(id(md)))
mod = types.ModuleType(
    'mod',
    'this is a module'

)
print(mod)




print('\n\n')

def funct_to_call():
    for i in sys.meta_path:
        return i
   


print(funct_to_call())
importer = sys.meta_path[0]