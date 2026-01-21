import { Box, Avatar, Paper, Typography } from '@mui/material'
import SmartToyIcon from '@mui/icons-material/SmartToy'
import PersonIcon from '@mui/icons-material/Person'
import ReactMarkdown from 'react-markdown'

interface ChatMessageProps {
  role: 'user' | 'assistant'
  content: string
}

export function ChatMessage({ role, content }: ChatMessageProps) {
  const isUser = role === 'user'

  return (
    <Box
      sx={{
        display: 'flex',
        gap: 2,
        mb: 2,
        flexDirection: isUser ? 'row-reverse' : 'row',
      }}
    >
      <Avatar
        sx={{
          bgcolor: isUser ? 'primary.main' : 'secondary.main',
          width: 40,
          height: 40,
        }}
      >
        {isUser ? <PersonIcon /> : <SmartToyIcon />}
      </Avatar>
      <Paper
        elevation={0}
        sx={{
          p: 2,
          maxWidth: '70%',
          bgcolor: isUser ? 'primary.dark' : 'background.paper',
          border: '1px solid',
          borderColor: isUser ? 'primary.main' : 'divider',
        }}
      >
        {isUser ? (
          <Typography
            sx={{
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {content}
          </Typography>
        ) : (
          <Box
            sx={{
              '& p': { m: 0, mb: 1 },
              '& p:last-child': { mb: 0 },
              '& pre': {
                bgcolor: 'background.default',
                p: 1.5,
                borderRadius: 1,
                overflow: 'auto',
              },
              '& code': {
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: '0.9em',
              },
            }}
          >
            <ReactMarkdown>{content}</ReactMarkdown>
          </Box>
        )}
      </Paper>
    </Box>
  )
}

