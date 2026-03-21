% Domain
file(1..8).
rank(1..8).

% Occupied squares (any piece or king)
occupied(X, Y) :- piece(_, _, X, Y).
occupied(X, Y) :- king(_, X, Y).

% -------------------------------------------------------
% Queen move generation (sliding)
% -------------------------------------------------------

% Horizontal blocking
blocked_h(FX, FY, TX) :-
    piece(_, _, BX, FY), FX < TX, BX > FX, BX < TX.
blocked_h(FX, FY, TX) :-
    king(_, BX, FY), FX < TX, BX > FX, BX < TX.
blocked_h(FX, FY, TX) :-
    piece(_, _, BX, FY), FX > TX, BX < FX, BX > TX.
blocked_h(FX, FY, TX) :-
    king(_, BX, FY), FX > TX, BX < FX, BX > TX.

% Vertical blocking
blocked_v(FX, FY, TY) :-
    piece(_, _, FX, BY), FY < TY, BY > FY, BY < TY.
blocked_v(FX, FY, TY) :-
    king(_, FX, BY), FY < TY, BY > FY, BY < TY.
blocked_v(FX, FY, TY) :-
    piece(_, _, FX, BY), FY > TY, BY < FY, BY > TY.
blocked_v(FX, FY, TY) :-
    king(_, FX, BY), FY > TY, BY < FY, BY > TY.

% Diagonal blocking helpers
% For diagonal moves: |TX-FX| = |TY-FY|
% We need to check if any piece lies strictly between (FX,FY) and (TX,TY) on the diagonal.

blocked_d(FX, FY, TX, TY) :-
    piece(_, _, BX, BY),
    FX < TX, FY < TY,
    BX > FX, BX < TX,
    BY > FY, BY < TY,
    BX - FX = BY - FY.
blocked_d(FX, FY, TX, TY) :-
    king(_, BX, BY),
    FX < TX, FY < TY,
    BX > FX, BX < TX,
    BY > FY, BY < TY,
    BX - FX = BY - FY.
blocked_d(FX, FY, TX, TY) :-
    piece(_, _, BX, BY),
    FX < TX, FY > TY,
    BX > FX, BX < TX,
    BY < FY, BY > TY,
    BX - FX = FY - BY.
blocked_d(FX, FY, TX, TY) :-
    king(_, BX, BY),
    FX < TX, FY > TY,
    BX > FX, BX < TX,
    BY < FY, BY > TY,
    BX - FX = FY - BY.
blocked_d(FX, FY, TX, TY) :-
    piece(_, _, BX, BY),
    FX > TX, FY < TY,
    BX < FX, BX > TX,
    BY > FY, BY < TY,
    FX - BX = BY - FY.
blocked_d(FX, FY, TX, TY) :-
    king(_, BX, BY),
    FX > TX, FY < TY,
    BX < FX, BX > TX,
    BY > FY, BY < TY,
    FX - BX = BY - FY.
blocked_d(FX, FY, TX, TY) :-
    piece(_, _, BX, BY),
    FX > TX, FY > TY,
    BX < FX, BX > TX,
    BY < FY, BY > TY,
    FX - BX = FY - BY.
blocked_d(FX, FY, TX, TY) :-
    king(_, BX, BY),
    FX > TX, FY > TY,
    BX < FX, BX > TX,
    BY < FY, BY > TY,
    FX - BX = FY - BY.

% Queen can move to (TX,TY) from (FX,FY) if path is clear and target not occupied by own piece
queen_can_move(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX = FX, TY != FY,
    not blocked_v(FX, FY, TY),
    not piece(white, _, TX, TY),
    not king(white, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TY = FY, TX != FX,
    not blocked_h(FX, FY, TX),
    not piece(white, _, TX, TY),
    not king(white, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX != FX, TY != FY,
    DX = TX - FX, DX >= 0,
    DY = TY - FY, DY >= 0,
    DX = DY,
    not blocked_d(FX, FY, TX, TY),
    not piece(white, _, TX, TY),
    not king(white, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX != FX, TY != FY,
    DX = FX - TX, DX >= 0,
    DY = FY - TY, DY >= 0,
    DX = DY,
    not blocked_d(FX, FY, TX, TY),
    not piece(white, _, TX, TY),
    not king(white, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX != FX, TY != FY,
    DX = TX - FX, DX >= 0,
    DY = FY - TY, DY >= 0,
    DX = DY,
    not blocked_d(FX, FY, TX, TY),
    not piece(white, _, TX, TY),
    not king(white, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX != FX, TY != FY,
    DX = FX - TX, DX >= 0,
    DY = TY - FY, DY >= 0,
    DX = DY,
    not blocked_d(FX, FY, TX, TY),
    not piece(white, _, TX, TY),
    not king(white, TX, TY).

% -------------------------------------------------------
% Squares attacked by white after queen moves to (QX,QY)
% -------------------------------------------------------

% White king attacks
king_attacks_white(TX, TY) :-
    king(white, KX, KY),
    file(TX), rank(TY),
    DX = TX - KX, DX >= 0,
    DY = TY - KY, DY >= 0,
    DX <= 1, DY <= 1,
    not (TX = KX, TY = KY).

king_attacks_white(TX, TY) :-
    king(white, KX, KY),
    file(TX), rank(TY),
    DX = KX - TX, DX >= 0,
    DY = TY - KY, DY >= 0,
    DX <= 1, DY <= 1,
    not (TX = KX, TY = KY).

king_attacks_white(TX, TY) :-
    king(white, KX, KY),
    file(TX), rank(TY),
    DX = TX - KX, DX >= 0,
    DY = KY - TY, DY >= 0,
    DX <= 1, DY <= 1,
    not (TX = KX, TY = KY).

king_attacks_white(TX, TY) :-
    king(white, KX, KY),
    file(TX), rank(TY),
    DX = KX - TX, DX >= 0,
    DY = KY - TY, DY >= 0,
    DX <= 1, DY <= 1,
    not (TX = KX, TY = KY).

% Queen attacks from new position (QX,QY) - horizontal
queen_attacks_h(QX, QY, TX, TY) :-
    file(TX), rank(TY),
    TY = QY, TX != QX,
    not blocked_h(QX, QY, TX).

% Queen attacks from new position (QX,QY) - vertical
queen_attacks_v(QX, QY, TX, TY) :-
    file(TX), rank(TY),
    TX = QX, TY != QY,
    not blocked_v(QX, QY, TY).

% Queen attacks from new position (QX,QY) - diagonal (all 4 directions)
queen_attacks_d(QX, QY, TX, TY) :-
    file(TX), rank(TY),
    TX != QX, TY != QY,
    DX = TX - QX, DX >= 0,
    DY = TY - QY, DY >= 0,
    DX = DY,
    not blocked_d(QX, QY, TX, TY).

queen_attacks_d(QX, QY, TX, TY) :-
    file(TX), rank(TY),
    TX != QX, TY != QY,
    DX = QX - TX, DX >= 0,
    DY = QY - TY, DY >= 0,
    DX = DY,
    not blocked_d(QX, QY, TX, TY).

queen_attacks_d(QX, QY, TX, TY) :-
    file(TX), rank(TY),
    TX != QX, TY != QY,
    DX = TX - QX, DX >= 0,
    DY = QY - TY, DY >= 0,
    DX = DY,
    not blocked_d(QX, QY, TX, TY).

queen_attacks_d(QX, QY, TX, TY) :-
    file(TX), rank(TY),
    TX != QX, TY != QY,
    DX = QX - TX, DX >= 0,
    DY = TY - QY, DY >= 0,
    DX = DY,
    not blocked_d(QX, QY, TX, TY).

% Combined queen attacks
queen_attacks(QX, QY, TX, TY) :- queen_attacks_h(QX, QY, TX, TY).
queen_attacks(QX, QY, TX, TY) :- queen_attacks_v(QX, QY, TX, TY).
queen_attacks(QX, QY, TX, TY) :- queen_attacks_d(QX, QY, TX, TY).

% -------------------------------------------------------
% After white queen moves from (FX,FY) to (QX,QY),
% is the black king in check?
% -------------------------------------------------------

% Occupied squares after the move (queen moved from FX,FY to QX,QY)
occupied_after(X, Y, FX, FY, QX, QY) :-
    piece(_, _, X, Y), not (X = FX, Y = FY), not (X = QX, Y = QY).
occupied_after(X, Y, FX, FY, QX, QY) :-
    king(_, X, Y), not (X = QX, Y = QY).
occupied_after(QX, QY, _, _, QX, QY).

% Blocking for queen at new position considering updated board
% We need to check if black king is attacked by queen at (QX,QY)
% The queen at (QX,QY) attacks (BKX,BKY) if path is clear on updated board.

% For the check detection, we use the queen_attacks predicate but need to
% account for the queen having moved. The blocked_h/v/d predicates use
% the original piece positions. We need versions that account for the move.

% Horizontal path clear after move
path_clear_h_after(QX, QY, TX, FX, FY) :-
    % Check no piece between QX and TX on rank QY, excluding original queen pos
    QX < TX,
    not blocked_h_after(QX, QY, TX, FX, FY).
path_clear_h_after(QX, QY, TX, FX, FY) :-
    QX > TX,
    not blocked_h_after(QX, QY, TX, FX, FY).

blocked_h_after(QX, QY, TX, FX, FY) :-
    piece(_, _, BX, QY),
    not (BX = FX, QY = FY),
    not (BX = QX, QY = QY),
    QX < TX, BX > QX, BX < TX.
blocked_h_after(QX, QY, TX, FX, FY) :-
    king(white, BX, QY),
    not (BX = QX),
    QX < TX, BX > QX, BX < TX.
blocked_h_after(QX, QY, TX, FX, FY) :-
    piece(_, _, BX, QY),
    not (BX = FX, QY = FY),
    not (BX = QX),
    QX > TX, BX < QX, BX > TX.
blocked_h_after(QX, QY, TX, FX, FY) :-
    king(white, BX, QY),
    not (BX = QX),
    QX > TX, BX < QX, BX > TX.

% Vertical path clear after move
blocked_v_after(QX, QY, TY, FX, FY) :-
    piece(_, _, QX, BY),
    not (QX = FX, BY = FY),
    not (BY = QY),
    QY < TY, BY > QY, BY < TY.
blocked_v_after(QX, QY, TY, FX, FY) :-
    king(white, QX, BY),
    not (BY = QY),
    QY < TY, BY > QY, BY < TY.
blocked_v_after(QX, QY, TY, FX, FY) :-
    piece(_, _, QX, BY),
    not (QX = FX, BY = FY),
    not (BY = QY),
    QY > TY, BY < QY, BY > TY.
blocked_v_after(QX, QY, TY, FX, FY) :-
    king(white, QX, BY),
    not (BY = QY),
    QY > TY, BY < QY, BY > TY.

% Diagonal path clear after move
blocked_d_after(QX, QY, TX, TY, FX, FY) :-
    piece(_, _, BX, BY),
    not (BX = FX, BY = FY),
    not (BX = QX, BY = QY),
    QX < TX, QY < TY,
    BX > QX, BX < TX,
    BY > QY, BY < TY,
    BX - QX = BY - QY.
blocked_d_after(QX, QY, TX, TY, FX, FY) :-
    king(white, BX, BY),
    not (BX = QX, BY = QY),
    QX < TX, QY < TY,
    BX > QX, BX < TX,
    BY > QY, BY < TY,
    BX - QX = BY - QY.
blocked_d_after(QX, QY, TX, TY, FX, FY) :-
    piece(_, _, BX, BY),
    not (BX = FX, BY = FY),
    not (BX = QX, BY = QY),
    QX < TX, QY > TY,
    BX > QX, BX < TX,
    BY < QY, BY > TY,
    BX - QX = QY - BY.
blocked_d_after(QX, QY, TX, TY, FX, FY) :-
    king(white, BX, BY),
    not (BX = QX, BY = QY),
    QX < TX, QY > TY,
    BX > QX, BX < TX,
    BY < QY, BY > TY,
    BX - QX = QY - BY.
blocked_d_after(QX, QY, TX, TY, FX, FY) :-
    piece(_, _, BX, BY),
    not (BX = FX, BY = FY),
    not (BX = QX, BY = QY),
    QX > TX, QY < TY,
    BX < QX, BX > TX,
    BY > QY, BY < TY,
    QX - BX = BY - QY.
blocked_d_after(QX, QY, TX, TY, FX, FY) :-
    king(white, BX, BY),
    not (BX = QX, BY = QY),
    QX > TX, QY < TY,
    BX < QX, BX > TX,
    BY > QY, BY < TY,
    QX - BX = BY - QY.
blocked_d_after(QX, QY, TX, TY, FX, FY) :-
    piece(_, _, BX, BY),
    not (BX = FX, BY = FY),
    not (BX = QX, BY = QY),
    QX > TX, QY > TY,
    BX < QX, BX > TX,
    BY < QY, BY > TY,
    QX - BX = QY - BY.
blocked_d_after(QX, QY, TX, TY, FX, FY) :-
    king(white, BX, BY),
    not (BX = QX, BY = QY),
    QX > TX, QY > TY,
    BX < QX, BX > TX,
    BY < QY, BY > TY,
    QX - BX = QY - BY.

% Queen at (QX,QY) attacks (TX,TY) after move from (FX,FY)
queen_attacks_after_h(QX, QY, TX, TY, FX, FY) :-
    file(TX), rank(TY),
    TY = QY, TX != QX,
    not blocked_h_after(QX, QY, TX, FX, FY).

queen_attacks_after_v(QX, QY, TX, TY, FX, FY) :-
    file(TX), rank(TY),
    TX = QX, TY != QY,
    not blocked_v_after(QX, QY, TY, FX, FY).

queen_attacks_after_d(QX, QY, TX, TY, FX, FY) :-
    file(TX), rank(TY),
    TX != QX, TY != QY,
    DX = TX - QX, DX >= 0,
    DY = TY - QY, DY >= 0,
    DX = DY,
    not blocked_d_after(QX, QY, TX, TY, FX, FY).

queen_attacks_after_d(QX, QY, TX, TY, FX, FY) :-
    file(TX), rank(TY),
    TX != QX, TY != QY,
    DX = QX - TX, DX >= 0,
    DY = QY - TY, DY >= 0,
    DX = DY,
    not blocked_d_after(QX, QY, TX, TY, FX, FY).

queen_attacks_after_d(QX, QY, TX, TY, FX, FY) :-
    file(TX), rank(TY),
    TX != QX, TY != QY,
    DX = TX - QX, DX >= 0,
    DY = QY - TY, DY >= 0,
    DX = DY,
    not blocked_d_after(QX, QY, TX, TY, FX, FY).

queen_attacks_after_d(QX, QY, TX, TY, FX, FY) :-
    file(TX), rank(TY),
    TX != QX, TY != QY,
    DX = QX - TX, DX >= 0,
    DY = TY - QY, DY >= 0,
    DX = DY,
    not blocked_d_after(QX, QY, TX, TY, FX, FY).

queen_attacks_after(QX, QY, TX, TY, FX, FY) :- queen_attacks_after_h(QX, QY, TX, TY, FX, FY).
queen_attacks_after(QX, QY, TX, TY, FX, FY) :- queen_attacks_after_v(QX, QY, TX, TY, FX, FY).
queen_attacks_after(QX, QY, TX, TY, FX, FY) :- queen_attacks_after_d(QX, QY, TX, TY, FX, FY).

% -------------------------------------------------------
% Check: is black king in check after queen moves to (QX,QY)?
% -------------------------------------------------------
in_check_after(QX, QY, FX, FY) :-
    king(black, BKX, BKY),
    queen_attacks_after(QX, QY, BKX, BKY, FX, FY).

in_check_after(QX, QY, FX, FY) :-
    king(black, BKX, BKY),
    king_attacks_white(BKX, BKY).

% -------------------------------------------------------
% Can black king escape to (TX,TY) after queen moves to (QX,QY)?
% -------------------------------------------------------

% A square is attacked by white queen at (QX,QY) after move from (FX,FY)
white_attacks_after(TX, TY, QX, QY, FX, FY) :-
    queen_attacks_after(QX, QY, TX, TY, FX, FY).

white_attacks_after(TX, TY, QX, QY, FX, FY) :-
    king_attacks_white(TX, TY).

% Black king can move to (TX,TY) if:
% 1. It's on the board
% 2. Not occupied by black piece
% 3. Not attacked by white after the move
% 4. Not the queen's square (or if it is, queen is not defended)

% Actually: black king captures queen only if queen square not defended by white king
% Queen square (QX,QY) is defended by white king if king_attacks_white(QX,QY)

% Black king escape square
bk_escape(TX, TY, QX, QY, FX, FY) :-
    king(black, BKX, BKY),
    file(TX), rank(TY),
    DX = TX - BKX, DX >= 0, DX <= 1,
    DY = TY - BKY, DY >= 0, DY <= 1,
    not (TX = BKX, TY = BKY),
    not king(black, TX, TY),
    not piece(black, _, TX, TY),
    not white_attacks_after(TX, TY, QX, QY, FX, FY),
    not (TX = QX, TY = QY).

bk_escape(TX, TY, QX, QY, FX, FY) :-
    king(black, BKX, BKY),
    file(TX), rank(TY),
    DX = BKX - TX, DX >= 0, DX <= 1,
    DY = TY - BKY, DY >= 0, DY <= 1,
    not (TX = BKX, TY = BKY),
    not king(black, TX, TY),
    not piece(black, _, TX, TY),
    not white_attacks_after(TX, TY, QX, QY, FX, FY),
    not (TX = QX, TY = QY).

bk_escape(TX, TY, QX, QY, FX, FY) :-
    king(black, BKX, BKY),
    file(TX), rank(TY),
    DX = TX - BKX, DX >= 0, DX <= 1,
    DY = BKY - TY, DY >= 0, DY <= 1,
    not (TX = BKX, TY = BKY),
    not king(black, TX, TY),
    not piece(black, _, TX, TY),
    not white_attacks_after(TX, TY, QX, QY, FX, FY),
    not (TX = QX, TY = QY).

bk_escape(TX, TY, QX, QY, FX, FY) :-
    king(black, BKX, BKY),
    file(TX), rank(TY),
    DX = BKX - TX, DX >= 0, DX <= 1,
    DY = BKY - TY, DY >= 0, DY <= 1,
    not (TX = BKX, TY = BKY),
    not king(black, TX, TY),
    not piece(black, _, TX, TY),
    not white_attacks_after(TX, TY, QX, QY, FX, FY),
    not (TX = QX, TY = QY).

% Black king can capture queen if queen square not defended
bk_escape(QX, QY, QX, QY, FX, FY) :-
    king(black, BKX, BKY),
    DX = QX - BKX, DX >= 0, DX <= 1,
    DY = QY - BKY, DY >= 0, DY <= 1,
    not (QX = BKX, QY = BKY),
    not king_attacks_white(QX, QY).

bk_escape(QX, QY, QX, QY, FX, FY) :-
    king(black, BKX, BKY),
    DX = BKX - QX, DX >= 0, DX <= 1,
    DY = QY - BKY, DY >= 0, DY <= 1,
    not (QX = BKX, QY = BKY),
    not king_attacks_white(QX, QY).

bk_escape(QX, QY, QX, QY, FX, FY) :-
    king(black, BKX, BKY),
    DX = QX - BKX, DX >= 0, DX <= 1,
    DY = BKY - QY, DY >= 0, DY <= 1,
    not (QX = BKX, QY = BKY),
    not king_attacks_white(QX, QY).

bk_escape(QX, QY, QX, QY, FX, FY) :-
    king(black, BKX, BKY),
    DX = BKX - QX, DX >= 0, DX <= 1,
    DY = BKY - QY, DY >= 0, DY <= 1,
    not (QX = BKX, QY = BKY),
    not king_attacks_white(QX, QY).

% -------------------------------------------------------
% Checkmate: in check and no escape
% -------------------------------------------------------
is_checkmate(FX, FY, QX, QY) :-
    queen_can_move(FX, FY, QX, QY),
    in_check_after(QX, QY, FX, FY),
    not bk_escape(_, _, QX, QY, FX, FY).

% -------------------------------------------------------
% Output
% -------------------------------------------------------
mate_move(white, queen, FX, FY, QX, QY) :-
    is_checkmate(FX, FY, QX, QY).

#show mate_move/6.