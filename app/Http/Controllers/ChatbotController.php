<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use GuzzleHttp\Client;

class ChatbotController extends Controller
{
    public function getResponse(Request $request)
    {
        $request->validate([
            'message' => 'required|string',
        ]);

        $client = new Client();
        $response = $client->post(env('VULTR_API_URL') . '/' . env('VULTR_MODEL_ID'), [
            'headers' => [
                'Authorization' => 'Bearer ' . env('VULTR_API_KEY'),
                'Content-Type' => 'application/json',
            ],
            'json' => [
                'prompt' => $request->message,
            ],
        ]);

        if ($response->getStatusCode() !== 200) {
            return response()->json(['error' => 'Failed to get response from the API'], 500);
        }

        $responseData = json_decode($response->getBody()->getContents(), true);

        return response()->json([
            'response' => $responseData['choices'][0]['text'] ?? 'No response available',
        ]);
    }
}
