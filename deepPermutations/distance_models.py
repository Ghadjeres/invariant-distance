import keras.backend as K
from keras.engine import Input
from keras.engine import Model

from keras.layers import Dense, Dropout, TimeDistributed, Activation, LSTM, \
    Lambda, RepeatVector, Reshape, Masking, Add, Concatenate


def l2_norm(y_true, y_pred):
    reg_lambda = 1.e-5
    return reg_lambda * K.sum(K.square(y_pred))


def l1_norm(y_true, y_pred):
    reg_lambda = 1.e-2
    return reg_lambda * K.sum(K.abs(y_pred))


def seq2seq_deep(timesteps, num_features, num_dense, num_units_lstm):
    input_seq = Input(shape=(timesteps, num_features), name='input_seq')

    # ---Encoding---
    # input dropout
    # predictions = Dropout(0.2)(input_seq)

    # embedding
    embedding = Dense(num_dense)
    predictions = TimeDistributed(embedding)(input_seq)

    # recurrent networks
    return_sequences = True
    for k, stack_index in enumerate(range(len(num_units_lstm))):
        if k == len(num_units_lstm) - 1:
            return_sequences = False

        if k > 0:
            # todo difference between concat and sum
            predictions_tmp = merge(
                [Activation('relu')(predictions), predictions_old], mode='sum')
        else:
            predictions_tmp = predictions

        predictions_old = predictions

        predictions = predictions_tmp

        predictions = LSTM(num_units_lstm[stack_index],
                           return_sequences=return_sequences,
                           name='lstm_encoding_' + str(stack_index)
                           )(predictions)

        predictions = Dropout(0.2)(predictions)

    # retain only last input for skip connections
    predictions_old = Lambda(lambda t: t[:, -1, :],
                             output_shape=lambda input_shape: (
                                 input_shape[0], input_shape[-1])
                             )(predictions_old)

    hidden_state = merge([Activation('relu')(predictions), predictions_old],
                         mode='sum')
    hidden_states = RepeatVector(timesteps)(hidden_state)

    # ---Decoding---
    # recurrent networks

    predictions = hidden_states
    return_sequences = True
    for k, stack_index in enumerate(range(len(num_units_lstm))):

        if k > 0:
            # todo difference between concat and sum
            predictions_tmp = merge(
                [Activation('relu')(predictions), predictions_old], mode='sum')
        else:
            predictions_tmp = predictions

        predictions_old = predictions

        predictions = predictions_tmp

        predictions = LSTM(num_units_lstm[stack_index],
                           return_sequences=return_sequences,
                           name='lstm_decoding_' + str(stack_index)
                           )(predictions)

        predictions = Dropout(0.2)(predictions)

    predictions = merge([Activation('relu')(predictions), predictions_old],
                        mode='sum')
    # ???
    predictions = TimeDistributed(Dense(num_features))(predictions)
    output_seq = Activation('sigmoid')(predictions)

    model = Model(input=input_seq,
                  output=output_seq)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def seq2seq(timesteps, num_features, num_units_lstm, dropout_prob=0.2,
            masking=False):
    input_seq = Input((timesteps, num_features), name='input_seq')

    if masking:
        output = Masking(mask_value=-1)(input_seq)
    else:
        output = input_seq

    output = Dropout(dropout_prob)(output)
    # encoding
    output = LSTM(num_units_lstm, return_sequences=False)(output)
    # output = LSTM(num_units_lstm, return_sequences=False)(output)
    output_reformat = Reshape((1, num_units_lstm))(output)
    # decoding
    # timesteps = 32

    # only infor at step one
    inputs_decoding = Lambda(lambda x: K.concatenate((
        K.concatenate([x, K.zeros_like(x)[:, :, 0:1]], axis=2),
        K.tile(
            K.concatenate([K.zeros_like(x), K.ones_like(x)[:, :, 0:1]],
                          axis=2),
            (1, 32 - 1, 1))
    ), axis=1))(output_reformat)
    # all info:
    # inputs_decoding = Lambda(lambda x: K.tile(x, (1, 16, 1)))(output_reformat)

    decoded = LSTM(num_units_lstm, return_sequences=True)(inputs_decoding)
    # decoded = LSTM(num_units_lstm, return_sequences=True)(decoded)
    preds = TimeDistributed(Dense(num_features))(decoded)
    preds = TimeDistributed(Activation('softmax'), name='output_seq')(preds)

    model = Model(input=input_seq, output=preds)

    model.compile(optimizer='rmsprop',
                  loss={'output_seq': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    return model


def invariant_seq2seq(timesteps, num_features, num_units_lstm, num_offsets,
                      dropout_prob=0.2, num_layers=1,
                      masking=False):
    # todo add skip connections
    input_seq = Input((timesteps, num_features), name='input_seq')
    input_offset = Input((num_offsets,), name='input_offset')

    if masking:
        output = Masking(mask_value=-1)(input_seq)
    else:
        output = input_seq

    if dropout_prob:
        output = Dropout(dropout_prob)(output)

    # encoding
    for i in range(num_layers - 1):
        output = LSTM(num_units_lstm, return_sequences=True)(output)
        output = Dropout(0.5)(output)
    output = LSTM(num_units_lstm, return_sequences=False,
                  name='hidden_repr')(output)

    # add offset
    output = merge([output, input_offset], mode='concat')
    output_reformat = Reshape((1, num_units_lstm + num_offsets))(output)

    # decoding
    # timesteps = 32

    # only info at step one
    inputs_decoding = Lambda(lambda x: K.concatenate((
        K.concatenate([x, K.zeros_like(x)[:, :, 0:1]], axis=2),
        K.tile(
            K.concatenate([K.zeros_like(x), K.ones_like(x)[:, :, 0:1]],
                          axis=2),
            (1, 32 - 1, 1))
    ), axis=1))(output_reformat)
    # all info:
    # inputs_decoding = Lambda(lambda x: K.tile(x, (1, 32, 1)))(output_reformat)

    decoded = inputs_decoding
    for i in range(num_layers):
        decoded = LSTM(num_units_lstm, return_sequences=True)(decoded)

    preds = TimeDistributed(Dense(num_features))(decoded)
    preds = TimeDistributed(Activation('softmax'), name='output_seq')(preds)

    model = Model(input=[input_seq, input_offset], output=preds)

    model.compile(optimizer='adam',
                  loss={'output_seq': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    return model


def invariant_absolute_seq2seq(timesteps, num_features, num_units_lstm,
                               num_pitches, dropout_prob=0.2, num_layers=1,
                               masking=False):
    # todo add skip connections
    input_seq = Input((timesteps, num_features), name='input_seq')
    input_offset = Input((num_pitches,), name='first_note')

    if masking:
        output = Masking(mask_value=-1)(input_seq)
    else:
        output = input_seq

    if dropout_prob:
        output = Dropout(dropout_prob)(output)

    # encoding
    for i in range(num_layers - 1):
        output = LSTM(num_units_lstm, return_sequences=True)(output)
        output = Dropout(0.5)(output)
    output = LSTM(num_units_lstm, return_sequences=False,
                  name='hidden_repr')(output)

    # add offset
    output = merge([output, input_offset], mode='concat')
    output_reformat = Reshape((1, num_units_lstm + num_pitches))(output)

    # decoding
    # timesteps = 32

    # only info at step one
    inputs_decoding = Lambda(lambda x: K.concatenate((
        K.concatenate([x, K.zeros_like(x)[:, :, 0:1]], axis=2),
        K.tile(
            K.concatenate([K.zeros_like(x), K.ones_like(x)[:, :, 0:1]],
                          axis=2),
            (1, 32 - 1, 1))
    ), axis=1))(output_reformat)
    # all info:
    # inputs_decoding = Lambda(lambda x: K.tile(x, (1, 32, 1)))(output_reformat)

    decoded = inputs_decoding
    for i in range(num_layers):
        decoded = LSTM(num_units_lstm, return_sequences=True)(decoded)

    preds = TimeDistributed(Dense(num_features))(decoded)
    preds = TimeDistributed(Activation('softmax'), name='output_seq')(preds)

    model = Model(input=[input_seq, input_offset], output=preds)

    model.compile(optimizer='adam',
                  loss={'output_seq': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    return model


def invariant_absolute_seq2seq_reg(timesteps, num_features, num_units_lstm,
                                   num_pitches, dropout_prob=0.2, num_layers=1,
                                   reg=l2_norm,
                                   masking=False):
    # todo add skip connections
    input_seq = Input((timesteps, num_features), name='input_seq')
    input_offset = Input((num_pitches,), name='first_note')
    transposed_input = Input((timesteps, num_features),
                             name='transposed_input')

    if masking:
        processed_input = Masking(mask_value=-1)(input_seq)
    else:
        processed_input = input_seq

    if dropout_prob:
        processed_input = Dropout(dropout_prob)(processed_input)

    # hidden_repr model
    hidden_repr_model_input = Input((timesteps, num_features))
    hidden_repr_model_output = hidden_repr_model_input
    for i in range(num_layers - 1):
        hidden_repr_model_output = LSTM(num_units_lstm, return_sequences=True)(
            hidden_repr_model_output)
        hidden_repr_model_output = Dropout(0.5)(hidden_repr_model_output)
    hidden_repr_model_output = LSTM(num_units_lstm, return_sequences=False)(
        hidden_repr_model_output)
    hidden_repr_model = Model(hidden_repr_model_input,
                              hidden_repr_model_output)

    processed_input = hidden_repr_model(processed_input)
    hidden_repr_input = Activation('linear', name='hidden_repr')(
        processed_input)

    # todo dropout?!
    hidden_repr_transposed = hidden_repr_model(transposed_input)
    hidden_repr_transposed = Lambda(lambda x: -1 * x)(hidden_repr_transposed)
    diff_hidden_repr = Add(name='diff_repr')(
        [hidden_repr_transposed, hidden_repr_input])

    # add offset
    output = Concatenate()([hidden_repr_input, input_offset])
    output_reformat = Reshape((1, num_units_lstm + num_pitches))(output)

    # decoding
    # timesteps = 32

    # only info at step one
    inputs_decoding = Lambda(lambda x: K.concatenate((
        K.concatenate([x, K.zeros_like(x)[:, :, 0:1]], axis=2),
        K.tile(
            K.concatenate([K.zeros_like(x), K.ones_like(x)[:, :, 0:1]],
                          axis=2),
            (1, 32 - 1, 1))
    ), axis=1))(output_reformat)
    # all info:
    # inputs_decoding = Lambda(lambda x: K.tile(x, (1, 32, 1)))(output_reformat)

    decoded = inputs_decoding
    for i in range(num_layers):
        decoded = LSTM(num_units_lstm, return_sequences=True)(decoded)

    preds = TimeDistributed(Dense(num_features))(decoded)
    preds = TimeDistributed(Activation('softmax'), name='output_seq')(preds)

    model = Model(input=[input_seq, input_offset, transposed_input],
                  output=[preds, diff_hidden_repr])

    model.compile(optimizer='adam',
                  loss={'output_seq': 'categorical_crossentropy',
                        'diff_repr': reg,
                        },
                  metrics=['accuracy'])
    return model


def invariant_absolute_seq2seq_reg_mean(timesteps, num_features,
                                        num_units_lstm, num_pitches,
                                        dropout_prob=0.2, num_layers=1,
                                        masking=False, reg=None):
    # todo add skip connections
    input_seq = Input((timesteps, num_features), name='input_seq')
    input_offset = Input((num_pitches,), name='first_note')
    transposed_input = Input((timesteps, num_features),
                             name='transposed_input')

    if masking:
        processed_input = Masking(mask_value=-1)(input_seq)
    else:
        processed_input = input_seq

    if dropout_prob:
        processed_input = Dropout(dropout_prob)(processed_input)

    # hidden_repr model
    hidden_repr_model_input = Input((timesteps, num_features))
    hidden_repr_model_output = hidden_repr_model_input
    for i in range(num_layers - 1):
        hidden_repr_model_output = LSTM(num_units_lstm, return_sequences=True)(
            hidden_repr_model_output)
        hidden_repr_model_output = Dropout(0.5)(hidden_repr_model_output)
    hidden_repr_model_output = LSTM(num_units_lstm, return_sequences=False)(
        hidden_repr_model_output)
    hidden_repr_model = Model(hidden_repr_model_input,
                              hidden_repr_model_output)

    # make invariant hidden_repr
    processed_input = hidden_repr_model(processed_input)
    hidden_repr_input = Activation('linear', name='hidden_repr')(
        processed_input)
    hidden_repr_transposed = hidden_repr_model(transposed_input)

    # sum
    sum_hidden_repr = Add(name='sum_repr')(
        [hidden_repr_transposed, hidden_repr_input])
    sum_hidden_repr = Lambda(lambda x: x / 2)(sum_hidden_repr)

    # diff
    neg_repr_transposed = Lambda(lambda x: -1 * x)(hidden_repr_transposed)
    diff_hidden_repr = Add(name='diff_hidden_repr')(
        [neg_repr_transposed, hidden_repr_input])

    # add offset
    output = Concatenate()([sum_hidden_repr, input_offset])
    output_reformat = Reshape((1, num_units_lstm + num_pitches))(output)

    # decoding
    # timesteps = 32

    # only info at step one
    inputs_decoding = Lambda(lambda x: K.concatenate((
        K.concatenate([x, K.zeros_like(x)[:, :, 0:1]], axis=2),
        K.tile(
            K.concatenate([K.zeros_like(x), K.ones_like(x)[:, :, 0:1]],
                          axis=2),
            (1, 32 - 1, 1))
    ), axis=1))(output_reformat)
    # all info:
    # inputs_decoding = Lambda(lambda x: K.tile(x, (1, 32, 1)))(output_reformat)

    decoded = inputs_decoding
    for i in range(num_layers):
        decoded = LSTM(num_units_lstm, return_sequences=True)(decoded)

    preds = TimeDistributed(Dense(num_features))(decoded)
    preds = TimeDistributed(Activation('softmax'), name='output_seq')(preds)

    model = Model(inputs=[input_seq, input_offset, transposed_input],
                  outputs=[preds, diff_hidden_repr])

    model.compile(optimizer='adam',
                  loss={'output_seq': 'categorical_crossentropy',
                        'diff_hidden_repr': reg},
                  # loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def invariant_absolute_seq2seq_reg_mean_relu(timesteps, num_features,
                                             num_units_lstm, num_pitches,
                                             dropout_prob=0.2, num_layers=1,
                                             masking=False, reg=None):
    # todo add skip connections
    input_seq = Input((timesteps, num_features), name='input_seq')
    input_offset = Input((num_pitches,), name='first_note')
    transposed_input = Input((timesteps, num_features),
                             name='transposed_input')

    if masking:
        processed_input = Masking(mask_value=-1)(input_seq)
    else:
        processed_input = input_seq

    if dropout_prob:
        processed_input = Dropout(dropout_prob)(processed_input)

    # hidden_repr model
    hidden_repr_model_input = Input((timesteps, num_features))
    hidden_repr_model_output = hidden_repr_model_input
    for i in range(num_layers - 1):
        hidden_repr_model_output = LSTM(num_units_lstm, return_sequences=True)(
            hidden_repr_model_output)
        hidden_repr_model_output = Dropout(0.5)(hidden_repr_model_output)
    hidden_repr_model_output = LSTM(num_units_lstm, return_sequences=False)(
        hidden_repr_model_output)
    hidden_repr_model_output = Activation('elu')(
        Dense(num_units_lstm * 2)(hidden_repr_model_output))
    hidden_repr_model = Model(hidden_repr_model_input,
                              hidden_repr_model_output)

    # must add a Dense layer before second LSTM
    intermediate_dense = Dense(num_units_lstm)

    # make invariant hidden_repr with relu
    processed_input = hidden_repr_model(processed_input)
    hidden_repr_input = Activation('linear', name='hidden_repr')(
        processed_input)
    hidden_repr_transposed = hidden_repr_model(transposed_input)

    # transform as dense non zero vector

    hidden_repr_input = intermediate_dense(hidden_repr_input)
    hidden_repr_transposed = intermediate_dense(hidden_repr_transposed)

    # sum
    sum_hidden_repr = Add(name='sum_repr')(
        [hidden_repr_transposed, hidden_repr_input])
    sum_hidden_repr = Lambda(lambda x: x / 2)(sum_hidden_repr)

    # diff
    neg_repr_transposed = Lambda(lambda x: -1 * x)(hidden_repr_transposed)
    diff_hidden_repr = Add(name='diff_hidden_repr')(
        [neg_repr_transposed, hidden_repr_input])

    # add offset
    output = Concatenate()([sum_hidden_repr, input_offset])
    output_reformat = Reshape((1, num_units_lstm + num_pitches))(output)

    # decoding
    # timesteps = 32

    # only info at step one
    inputs_decoding = Lambda(lambda x: K.concatenate((
        K.concatenate([x, K.zeros_like(x)[:, :, 0:1]], axis=2),
        K.tile(
            K.concatenate([K.zeros_like(x), K.ones_like(x)[:, :, 0:1]],
                          axis=2),
            (1, 32 - 1, 1))
    ), axis=1))(output_reformat)
    # all info:
    # inputs_decoding = Lambda(lambda x: K.tile(x, (1, 32, 1)))(output_reformat)

    decoded = inputs_decoding
    for i in range(num_layers):
        decoded = LSTM(num_units_lstm, return_sequences=True)(decoded)

    preds = TimeDistributed(Dense(num_features))(decoded)
    preds = TimeDistributed(Activation('softmax'), name='output_seq')(preds)

    model = Model(inputs=[input_seq, input_offset, transposed_input],
                  outputs=[preds, diff_hidden_repr])

    model.compile(optimizer='adam',
                  loss={'output_seq': 'categorical_crossentropy',
                        'diff_hidden_repr': reg},
                  # loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def invariant_seq2seq_NN(timesteps, num_features, num_units_lstm, num_offsets,
                         dropout_prob=0.2, masking=False):
    num_dense = 1024
    input_seq = Input((timesteps, num_features), name='input_seq')
    input_offset = Input((num_offsets,), name='input_offset')

    if masking:
        output = Masking(mask_value=-1)(input_seq)
    else:
        output = input_seq

    if dropout_prob:
        output = Dropout(dropout_prob)(output)
    # encoding
    # output = LSTM(num_units_lstm, return_sequences=True)(output)
    output = LSTM(num_units_lstm, return_sequences=False,
                  activation='sigmoid')(output)

    # NN output
    # output = Dense(num_dense)(output)
    output = Activation('linear', name='hidden_repr')(output)

    # compute offset
    # offset = Dense(num_dense)(input_offset)
    # offset = Activation('relu')(offset)
    # offset = Dense(num_dense)(offset)

    # add offset
    # output = merge([output, offset], mode='concat')

    # output_reformat = Reshape((1, num_dense))(output)
    output_reformat = Reshape((1, num_units_lstm))(output)

    # decoding
    # timesteps = 32

    # only infor at step one
    inputs_decoding = Lambda(lambda x: K.concatenate((
        K.concatenate([x, K.zeros_like(x)[:, :, 0:1]], axis=2),
        K.tile(
            K.concatenate([K.zeros_like(x), K.ones_like(x)[:, :, 0:1]],
                          axis=2),
            (1, 32 - 1, 1))
    ), axis=1))(output_reformat)
    # all info:
    # inputs_decoding = Lambda(lambda x: K.tile(x, (1, 32, 1)))(output_reformat)

    decoded = LSTM(num_units_lstm, return_sequences=True)(inputs_decoding)
    # decoded = LSTM(num_units_lstm, return_sequences=True)(decoded)

    preds = TimeDistributed(Dense(num_features))(decoded)
    preds = TimeDistributed(Activation('softmax'), name='output_seq')(preds)

    model = Model(input=[input_seq, input_offset], output=preds)

    model.compile(optimizer='adam',
                  loss={'output_seq': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    return model
