# web-mind-assistant-api

Crea tu entorno virtual con:

``` shell
C:\Python\Python312\python.exe -m venv venv

python -m venv venv
```

Activa tu entorno virtual en window con:

``` shell
venv\Scripts\activate
```

Puede llegar a ver errores al iniciar el entorno en windows, lo que se puede hacer es ejecutar el siguiente comando:

``` shell
Set-ExecutionPolicy RemoteSigned
```

Apartir de aquí ya puedes instalar paquetes normalmente con **pip**

### NOTAS:

1. instala keras, keras_preprocessing y tensorflow para no tener fallas

pip freeze > requirements.txt