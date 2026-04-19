import gradio as gr

from about import ABOUT_TEXT, INTRO_TEXT, SUBMISSION_TEXT, TITLE
from utils import load_results_dataframe, submit_zip


def refresh_leaderboard():
    return load_results_dataframe()


with gr.Blocks(title="The Well Leaderboard MVP") as demo:
    gr.Markdown(TITLE)
    gr.Markdown(INTRO_TEXT)

    with gr.Tab("Leaderboard"):
        leaderboard = gr.Dataframe(
            value=refresh_leaderboard,
            label="Ranked results",
            wrap=True,
        )
        refresh_button = gr.Button("Refresh leaderboard")
        refresh_button.click(fn=refresh_leaderboard, outputs=leaderboard)

    with gr.Tab("Submit"):
        gr.Markdown(SUBMISSION_TEXT)
        zip_file = gr.File(label="Submission zip", file_count="single", file_types=[".zip"])
        submit_button = gr.Button("Submit")
        submission_status = gr.Markdown()
        submit_button.click(fn=submit_zip, inputs=zip_file, outputs=submission_status)

    with gr.Tab("About"):
        gr.Markdown(ABOUT_TEXT)

demo.launch()
