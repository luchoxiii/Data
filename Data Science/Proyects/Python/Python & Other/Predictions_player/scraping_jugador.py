import pandas as pd
import requests
from lxml import html
import os

# Definir el rango de años
start_year = 2021
end_year = 2024  

# URL base sin el año
base_url = 'https://espndeportes.espn.com/basquetbol/nba/jugador/juego-a-juego/_/id/3975/tipo/nba/ano/'

# Definir headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
}

# Crear una lista para almacenar todos los datos
all_data = []

for year in range(start_year, end_year + 1):
    # Construir la URL para el año actual
    url = f'{base_url}{year}'
    
    print(f"Descargando datos para el año {year}...")

    # Hacer la solicitud HTTP con headers
    response = requests.get(url, headers=headers)

    # Verifica que la solicitud fue exitosa
    if response.status_code == 200:
        tree = html.fromstring(response.content)

        # XPath para el contenedor que contiene las tablas
        container_xpath = '//*[@id="fittPageContainer"]/div[2]/div/div[5]/div/div[1]/div[1]/div/div[2]'

        # Obtener el contenedor
        container = tree.xpath(container_xpath)

        if container:
            # Encontrar todas las tablas dentro del contenedor
            tables = container[0].xpath('.//table')

            for i, table in enumerate(tables, start=1):
                # Convertir cada tabla a un formato legible (como lista de filas)
                rows = table.xpath('.//tr')

                for row in rows:
                    cells = []  # Creamos una lista vacía para almacenar los valores de la fila

                    # Iterar sobre las celdas de la fila
                    for idx, cell in enumerate(row.xpath('.//td')):  # Usamos enumerate para saber la posición
                        value = cell.xpath('.//text()')  # Obtener el texto de la celda

                        if value:  # Si hay texto en la celda
                            value = value[0].strip()  # Limpia el texto
                            
                            # Si el texto tiene un enlace dentro de la celda, trata de extraer el texto dentro del enlace
                            if idx == 1:  # Segundo elemento (por ejemplo, nombre del jugador)
                                value = cell.xpath('.//span/span[3]/a/text()')
                                value = value[0].strip() if value else value
                            elif idx == 2:  # Tercer elemento (por ejemplo, detalles adicionales)
                                value1 = cell.xpath('.//a/div/span/text()')
                                value2 = cell.xpath('.//a/div/div/div/text()')
                                if value1 and value2:
                                    value = [value1[0].strip(), value2[0].strip()]  # Extrae ambos valores
                                else:
                                    value = value[0].strip()

                            if value:
                                cells.append(value)

                    if cells:  # Asegurarse de que hay celdas en la fila
                        # Agregar el año al principio de la fila
                        cells.insert(0, year)  
                        all_data.append(cells)  # Agregar la fila a la lista total
                        
            print(f"Tabla {i} del año {year} procesada.")
        else:
            print("Contenedor no encontrado.")
    else:
        print(f"Error al acceder a la página para el año {year}: {response.status_code}")

# Verifica si el directorio existe y lo crea si no existe
directory = r'C:\Users\Lucho\Documents'  # Cambia esto por una ruta válida
if not os.path.exists(directory):
    os.makedirs(directory)

# Definir la ruta del archivo CSV
csv_file = os.path.join(directory, 'nba_juego_a_juego.csv')

# Sobrescribir el archivo si ya existe
if os.path.exists(csv_file):
    os.remove(csv_file)

# Convertir la lista de datos a un DataFrame de pandas
df = pd.DataFrame(all_data)

# Verifica cuántas columnas tienes
num_columns = len(df.columns)

# Generar encabezados adecuados (asumimos que el número de columnas es consistente)
csv_headers = ['Año'] + [f'Columna {i+1}' for i in range(num_columns - 1)]

# Guardar el DataFrame en un archivo CSV
df.to_csv(csv_file, index=False, header=csv_headers)

print(f"Datos guardados en {csv_file}")