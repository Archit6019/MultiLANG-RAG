import gradio as gr
from main import DocumentProcessor 
import subprocess
import os
import time

processor = DocumentProcessor()

def setup_qdrant():
    try:
        print(f"Pulling Qdrant image...")
        subprocess.run(["docker", "pull", "qdrant/qdrant"], check=True)

        storage_path = os.path.join(os.getcwd(), "qdrant_storage")
        os.makedirs(storage_path, exist_ok=True)

        subprocess.run(["docker", "rm", "-f", "qdrant_container"], capture_output=True)
        print("Starting Qdrant container...")
        subprocess.run([
            "docker", "run",
            "-d",
            "--name", "qdrant_container",
            "-p", "6333:6333",
            "-p", "6334:6334",
            "-v", f"{storage_path}:/qdrant/storage:z",
            "qdrant/qdrant"
        ], check=True)

        time.sleep(5)

        result = subprocess.run(
            ["docker", "ps", "-q", "-f", "name=qdrant_container"],
            capture_output=True,
            text=True
        )

        print(f"Qdrant container status: {bool(result.stdout.strip())}")
        return bool(result.stdout.strip())
    except Exception as e:
        print(f"Error setting up Qdrant: {e}")
        return False

def ui():

    qdrant_ready = setup_qdrant()

    with gr.Blocks() as app:
        gr.Markdown("## AI-Driven Document and Chat Interface")

        if not qdrant_ready:
            gr.Markdown("## Qdrant container failed to start. Please check the logs for more details.")
            return

        with gr.Tab("Create Collection"):
            name = gr.Textbox(label="Collection Name")
            vector_size = gr.Number(label="Vector Size")
            create_button = gr.Button("Create Collection")
            create_output = gr.Textbox(label="Response")

            def handle_create_collection(name, vector_size):
                response = processor.create_collection(name, int(vector_size))
                return response if isinstance(response, str) else str(response)

            create_button.click(
                handle_create_collection,
                inputs=[name, vector_size],
                outputs=[create_output],
            )

        with gr.Tab("Upload Document"):
            collection_name = gr.Textbox(label="Collection Name")
            file = gr.File(label="Upload Document")
            file_name = gr.Textbox(label="Document Name")
            upload_button = gr.Button("Upload Document")
            upload_output = gr.Textbox(label="Response")

            def handle_upload_document(collection_name, file, file_name, doc_type):
                if file is None:
                    return "Please upload a file."
                response = processor.upload_document(
                    collection_name, file.name, file_name, doc_type
                )
                return response if isinstance(response, str) else str(response)

            upload_button.click(
                handle_upload_document,
                inputs=[collection_name, file, file_name],
                outputs=[upload_output],
            )

        with gr.Tab("Chat"):
            chat_collection_name = gr.Textbox(label="Collection Name")
            chatbot = gr.Chatbot(label="Chat History")
            user_message = gr.Textbox(label="Your Message")
            chat_button = gr.Button("Ask")
            clear_button = gr.Button("Clear History")
            ai_response_output = gr.Textbox(label="AI Response", lines=3)
            search_results_output = gr.Textbox(label="Search Results", lines=5)

            def handle_chat(collection_name, message, history):
                response = processor.chat(collection_name, message)
                if isinstance(response, dict):
                    ai_response = response.get("ai_response", "No AI response available.")
                    search_results = response.get("search_results", [])

                    unique_results = {}
                    for item in search_results:
                        doc_id = item.get("document_id")
                        if doc_id and doc_id not in unique_results:
                            unique_results[doc_id] = item

                    filtered_results = list(unique_results.values())
                    formatted_results = "\n".join(
                        f"{i + 1}. {item['document_name']}"
                        for i, item in enumerate(filtered_results)
                    )
                    
                    history.append((message, ai_response))
                    return history, "", formatted_results or "No search results found."
                else:
                    history.append((message, str(response)))
                    return history, "", "Error retrieving search results."

            def clear_history():
                processor.context = [processor.context[0]]
                return None, "", ""

            clear_button.click(
                clear_history,
                outputs=[chatbot, user_message, search_results_output],
            )

            chat_button.click(
                handle_chat,
                inputs=[chat_collection_name, user_message, chatbot],
                outputs=[chatbot,user_message, search_results_output],
            )

    return app

if __name__ == "__main__":
    ui().launch()
