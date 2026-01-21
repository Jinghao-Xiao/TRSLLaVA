def image_query_prompt() -> str:
    return (
        "I'm sending you a satellite/aerial image. Please analyze this image thoroughly. "
        "Assume the top of the image represents the North, the bottom is South, the left is West, and the right is East. "
        "Identify the main objects and describe their positions within the image. "
        "Regarding the proportion of objects, do NOT give exact percentages; use qualitative size terms. "
        "Focus on what's actually visible in the image."
    )

