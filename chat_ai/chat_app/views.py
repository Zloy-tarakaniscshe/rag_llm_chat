from rag_system.test import RAGSystem
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import Message


@csrf_exempt
def chat_view(request):
    if request.method == "GET":
        messages = Message.objects.order_by("created_at").all()[:10]
        context = {"messages": messages}
        return render(request, "base.html", context)

    elif request.method == "POST":
        user_input = request.POST.get("user_input")
        if not user_input:
            return JsonResponse({"error": "Empty input"}, status=400)

        message = Message(user_input=user_input)
        message.save()

        rag = RAGSystem()
        response = rag.get_responce(user_input)

        message.ai_response = response
        message.save()
        return JsonResponse({"ai_response": message.ai_response}, status=200)
