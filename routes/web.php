<?php

use App\Http\Controllers\WeatherController;
use App\Http\Controllers\ChatbotController;
use Illuminate\Support\Facades\Route;

Route::get('/', [WeatherController::class, 'index'])->name('home');
Route::post('/predict', [WeatherController::class, 'predict'])->name('predict');
Route::post('/predict-image', [WeatherController::class, 'predictImage'])->name('predict.image');
Route::post('/chat', [ChatbotController::class, 'chat'])->name('chat');
Route::get('/chatbot', [ChatbotController::class, 'index'])->name('chatbot.index');
Route::post('/chatbot/response', [ChatbotController::class, 'getResponse'])->name('chatbot.response');
