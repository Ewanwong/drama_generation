with open("pipeline/100_w_tfidf_outlines.txt", 'r', encoding='utf-8') as f:
    with open("tfidf_outline.txt", 'w', encoding='utf-8') as f1:
        text = f.read()
        generated = text.split("GENERATED_SCENE:")
        for raw in generated[1:]:
            generated_scene = raw.split("----------")[0]
            f1.write(generated_scene)
            f1.write('-------------')

    with open("start_model4.txt", 'w', encoding='utf-8') as f2:

        starts = text.split("START_OF_SCENE:")
        for raw in starts[1:]:
            start = raw.split('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')[0]
            f2.write(start)
            f2.write('^^^^^^^^^')
