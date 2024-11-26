# Cargar las librerías necesarias
library(MASS)       
library(caret)      
library(Metrics)     
library(ggplot2)

# Generación de la semilla
set.seed(42)

# Generación de los datos
tamanio_m2 <- sample(50:300, 100, replace = TRUE)  
precio <- tamanio_m2 * 2000 + sample(-100000:10000, 100, replace = TRUE)

data <- data.frame(tamanio_m2, precio)

# Separar variables
X <- data[, "tamanio_m2", drop = FALSE]
y <- data[, "precio", drop = FALSE]

# Separar en conjunto de entrenamiento y prueba
trainIndex <- createDataPartition(y$precio, p = 0.8, list = FALSE)
X_train <- X[trainIndex, , drop = FALSE]
y_train <- y[trainIndex, , drop = FALSE]
X_test <- X[-trainIndex, , drop = FALSE]
y_test <- y[-trainIndex, , drop = FALSE]

# Crear y entrenar el modelo
modelo <- lm(precio ~ tamanio_m2, data = data[trainIndex, , drop = FALSE])

# Realizar predicciones
predicciones <- predict(modelo, X_test)

# Calcular el error cuadrático medio (RMSE)
error <- mean((y_test$precio - predicciones)^2)
r_error <- sqrt(error)

# Imprimir resultados
cat("Error cuadrático es:", error, "\n")
cat("Raíz del error cuadrático es:", r_error, "\n")


# Graficar los datos reales y la línea de regresión
ggplot(data, aes(x = tamanio_m2, y = precio)) +
  geom_point(color = "green", alpha = 0.7, aes(label = "Datos reales")) +  # Datos reales
  geom_smooth(method = "lm", color = "red", aes(label = "Línea de regresión"), se = FALSE) +  # Línea de regresión
  labs(
    x = "Tamaño en m²",
    y = "Precio (U$D)",
    title = "Relación Tamaño & Precio"
  ) +
  theme_minimal() +
  theme(legend.position = "top") +
  scale_color_manual(values = c("Datos reales" = "green", "Línea de regresión" = "red"))