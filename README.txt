#Instituto Tecnológico de Costa Rica
#Escuela de Ingeniería Electrónica
#Prof: Pablo Alvarado Moya
#Tarea 2 IRP
#Jimena Salas Alfaro, carné 2016085746
#Roberto Gutiérrez Sánchez, carné 2016134351
#24 de Agosto, 2019

Instrucciones para ejecutar el script:

1. Ejecutar desde terminal los comandos:
	
	(opcional si ya cuenta con el paquete)
	$ apt install octave-optim
	
	$ octave --persist gradient_descent.m

Nota: Es importante que se ejecute la GUI de Octave, ya que el paquete de qt utilizado
para la interfaz gráfica de la tarea no está disponible a través de la línea de comandos de Octave.

2. Una vez que se abra la Figura 1, deberá seleccionar por medio de las barras deslizables los valores 
para el punto de inicio del algoritmo, o si lo desea, dejar los valores que están seleccionados por defecto.

3. Para iniciar la ejecución del algoritmo, debe hacer click en el botón "Calculate Gradient".

4. Se le presentarán las figuras correspondientes al error, evolución de la hipótesis y trayectoria de los
algoritmos de descenso de gradiente por lote y estocástico.

5. Si desea probar otro punto de partida, únicamente debe repetir los pasos 2 y 3 en la Figura 1, sin
necesidad de cerrar y abrir el programa de nuevo.
