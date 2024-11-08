<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Symfony\Component\Process\Process;
use Illuminate\Support\Facades\Storage;

class WeatherController extends Controller
{
    public function index()
    {
        return view('weather.index');
    }

    public function predict(Request $request)
    {
        $request->validate([
            'city' => 'required|string',
            'duration' => 'required|in:30,180,365,730'
        ]);

        $process = new Process([
            env('PYTHON_PATH'),
            base_path('python/Backend.py'),
            $request->city,
            $request->duration
        ]);

        $process->run();

        if (!$process->isSuccessful()) {
            return response()->json([
                'error' => 'Prediction failed'
            ], 500);
        }

        return response()->json([
            'data' => json_decode($process->getOutput(), true)
        ]);
    }

    public function predictImage(Request $request)
    {
        $request->validate([
            'image' => 'required|image|max:2048'
        ]);

        $path = $request->file('image')->store('temp');

        $process = new Process([
            env('PYTHON_PATH'),
            base_path('python/DL.py'),
            Storage::path($path)
        ]);

        $process->run();

        Storage::delete($path);

        if (!$process->isSuccessful()) {
            return response()->json([
                'error' => 'Image prediction failed'
            ], 500);
        }

        return response()->json([
            'data' => json_decode($process->getOutput(), true)
        ]);
    }
}